"""Persistent experiment queue.

A crash-safe queue backed by three append-only JSONL files plus an atomic
``pending.json`` snapshot:

    state/
      pending.json   – list of pending ExperimentSpec dicts (atomic rewrite)
      inflight.jsonl – append-only log of pops with epoch ms; the loop reconciles
                       this against done.jsonl at startup to recover crashes
      done.jsonl     – append-only log of completed experiments + result summary
      failed.jsonl   – append-only log of permanently failed experiments

Why this shape:
  * append-only means no concurrent-writer races even across restarts
  * a snapshotted pending list is cheap to rewrite (we have ~thousands of entries)
  * reconciliation at startup is trivial: anything in inflight but not in done|failed
    is requeued (the previous run was killed mid-experiment)

The queue is never empty. When the pending list runs out, :meth:`populate_seeds`
re-seeds it from the current registry — so the loop keeps experimenting forever.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("autoresearch.queue")


@dataclass
class ExperimentSpec:
    id: str
    type: str
    params: dict[str, Any]
    parent_id: str | None = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, type: str, params: dict[str, Any], parent_id: str | None = None) -> "ExperimentSpec":
        # Deterministic ID so identical specs don't pile up in the queue.
        payload = json.dumps({"type": type, "params": params}, sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
        rand = uuid.uuid4().hex[:4]
        return cls(id=f"{type}-{digest}-{rand}", type=type, params=params, parent_id=parent_id)


class Queue:
    """File-backed, restart-safe experiment queue.

    Single-process; the loop is the sole writer. The lock guards in-memory state
    only — concurrent processes would corrupt the files anyway and aren't
    supported.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.state_dir / "pending.json"
        self.inflight_path = self.state_dir / "inflight.jsonl"
        self.done_path = self.state_dir / "done.jsonl"
        self.failed_path = self.state_dir / "failed.jsonl"
        self._lock = threading.Lock()
        self._pending: list[ExperimentSpec] = []
        self._seen_ids: set[str] = set()
        self._load()
        self._reconcile_crash()

    # ---- io -----------------------------------------------------------------

    def _load(self) -> None:
        if self.pending_path.is_file():
            try:
                with open(self.pending_path) as f:
                    raw = json.load(f)
                self._pending = [ExperimentSpec(**r) for r in raw]
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("pending.json corrupt (%s); starting empty.", e)
                self._pending = []
        # Build seen set from pending + done + failed so dedup survives restarts.
        self._seen_ids = {e.id for e in self._pending}
        for path in (self.done_path, self.failed_path):
            if path.is_file():
                with open(path) as f:
                    for line in f:
                        try:
                            self._seen_ids.add(json.loads(line)["id"])
                        except (json.JSONDecodeError, KeyError):
                            continue

    def _persist_pending(self) -> None:
        """Atomic-ish: write to .tmp then rename."""
        tmp = self.pending_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump([asdict(e) for e in self._pending], f)
        os.replace(tmp, self.pending_path)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _reconcile_crash(self) -> None:
        """Anything in inflight but not in done|failed = a crash; requeue it."""
        if not self.inflight_path.is_file():
            return
        terminal: set[str] = set()
        for path in (self.done_path, self.failed_path):
            if path.is_file():
                with open(path) as f:
                    for line in f:
                        try:
                            terminal.add(json.loads(line)["id"])
                        except (json.JSONDecodeError, KeyError):
                            continue
        recovered: list[ExperimentSpec] = []
        with open(self.inflight_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec["id"] in terminal:
                        continue
                    spec = ExperimentSpec(
                        id=rec["id"],
                        type=rec["type"],
                        params=rec["params"],
                        parent_id=rec.get("parent_id"),
                        created_at=rec.get("created_at", time.time()),
                    )
                    if spec.id not in {e.id for e in self._pending}:
                        recovered.append(spec)
                except (json.JSONDecodeError, KeyError):
                    continue
        if recovered:
            log.info("queue: recovered %d crashed experiments", len(recovered))
            self._pending.extend(recovered)
            self._persist_pending()
        # Truncate the inflight log now that we've reconciled.
        self.inflight_path.write_text("")

    # ---- api ----------------------------------------------------------------

    def push(self, spec: ExperimentSpec) -> bool:
        with self._lock:
            if spec.id in self._seen_ids:
                return False
            self._pending.append(spec)
            self._seen_ids.add(spec.id)
            self._persist_pending()
            return True

    def push_front(self, spec: ExperimentSpec) -> bool:
        """Queue at the head — for high-priority specs like LLM proposer
        triggers that should run before the canonical backlog is drained."""
        with self._lock:
            if spec.id in self._seen_ids:
                return False
            self._pending.insert(0, spec)
            self._seen_ids.add(spec.id)
            self._persist_pending()
            return True

    def push_many(self, specs: Iterable[ExperimentSpec]) -> int:
        added = 0
        with self._lock:
            for spec in specs:
                if spec.id in self._seen_ids:
                    continue
                self._pending.append(spec)
                self._seen_ids.add(spec.id)
                added += 1
            if added:
                self._persist_pending()
        return added

    def pop(self) -> ExperimentSpec | None:
        with self._lock:
            if not self._pending:
                return None
            spec = self._pending.pop(0)
            self._persist_pending()
        self._append_jsonl(self.inflight_path, asdict(spec))
        return spec

    def record_done(self, spec: ExperimentSpec, result: dict[str, Any]) -> None:
        self._append_jsonl(
            self.done_path,
            {**asdict(spec), "finished_at": time.time(), "result": result},
        )

    def record_failed(self, spec: ExperimentSpec, error: str) -> None:
        self._append_jsonl(
            self.failed_path,
            {**asdict(spec), "finished_at": time.time(), "error": error},
        )

    # ---- stats --------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        def _count_lines(path: Path) -> int:
            if not path.is_file():
                return 0
            with open(path) as f:
                return sum(1 for _ in f)

        return {
            "pending": len(self._pending),
            "done": _count_lines(self.done_path),
            "failed": _count_lines(self.failed_path),
        }

    def is_empty(self) -> bool:
        with self._lock:
            return not self._pending


__all__ = ["ExperimentSpec", "Queue"]
