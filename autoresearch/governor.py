"""Resource governor for the autoresearch loop.

Enforces three hard constraints from the original spec:
  * CPU only (assertion at startup; we never allocate a device)
  * Total RSS for the loop process tree ≤ ``ram_cap_bytes`` (default 30 GB)
  * State directory size ≤ ``disk_cap_bytes`` (default 10 GB)

Strategy
--------
The governor is consulted between experiments and (cheaply) inside long
experiments via :func:`Governor.check`. It never kills the loop directly;
callers decide what to do. The two signals are:

* ``Pressure.NONE``  — fine to run.
* ``Pressure.SOFT``  — slow down: sleep, GC, rotate old shards.
* ``Pressure.HARD``  — abort the in-flight experiment; the loop wrapper will
  restart the process from clean state.

This split keeps fail-stop behaviour cheap (no try/except acrobatics inside
experiment code) while still bounding worst-case footprint.
"""
from __future__ import annotations

import enum
import gc
import gzip
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

log = logging.getLogger("autoresearch.governor")

GIB = 1024 ** 3


class Pressure(enum.IntEnum):
    NONE = 0
    SOFT = 1
    HARD = 2


@dataclass(frozen=True)
class GovernorConfig:
    ram_cap_bytes: int = 30 * GIB
    ram_soft_frac: float = 0.80          # back-pressure once 80% of cap is used
    disk_cap_bytes: int = 10 * GIB
    disk_soft_frac: float = 0.80
    rotate_keep_bytes: int = 4 * GIB     # after rotation, results dir should be ≤ this
    poll_interval_s: float = 1.0


class Governor:
    """Per-process governor. Cheap to call (``check()`` is just two psutil reads)."""

    def __init__(self, state_dir: Path | str, config: GovernorConfig | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.results_dir = self.state_dir / "results"
        self.archive_dir = self.state_dir / "archive"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or GovernorConfig()
        self.proc = psutil.Process()
        self._ensure_cpu_only()

    # ---- assertions ----------------------------------------------------------

    @staticmethod
    def _ensure_cpu_only() -> None:
        """Refuse to start under env that would bring a GPU/TPU into scope."""
        # We don't import torch.cuda here to keep startup light; we just refuse
        # if the env has been set to advertise a device.
        bad_env = [k for k in ("CUDA_VISIBLE_DEVICES", "TPU_NAME") if os.environ.get(k)]
        if bad_env:
            log.warning("Governor: %s set; forcing CPU-only by clearing them.", bad_env)
            for k in bad_env:
                os.environ.pop(k, None)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # ---- measurement ---------------------------------------------------------

    def ram_used_bytes(self) -> int:
        """RSS of the loop process and its children (workers, if any)."""
        try:
            total = self.proc.memory_info().rss
            for c in self.proc.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except psutil.NoSuchProcess:
            return 0

    def disk_used_bytes(self) -> int:
        """Bytes occupied by the autoresearch state dir (results + archive + meta)."""
        total = 0
        for root, _dirs, files in os.walk(self.state_dir):
            for name in files:
                fp = Path(root) / name
                try:
                    total += fp.stat().st_size
                except FileNotFoundError:
                    pass
        return total

    # ---- decisions -----------------------------------------------------------

    def check(self) -> Pressure:
        cfg = self.config
        ram = self.ram_used_bytes()
        disk = self.disk_used_bytes()
        if ram >= cfg.ram_cap_bytes or disk >= cfg.disk_cap_bytes:
            log.warning("Governor HARD: ram=%.2f GiB disk=%.2f GiB", ram / GIB, disk / GIB)
            return Pressure.HARD
        if ram >= cfg.ram_cap_bytes * cfg.ram_soft_frac or disk >= cfg.disk_cap_bytes * cfg.disk_soft_frac:
            return Pressure.SOFT
        return Pressure.NONE

    # ---- mitigation ----------------------------------------------------------

    def relieve(self) -> None:
        """Run on SOFT pressure. Cheap mitigations first, then rotation."""
        gc.collect()
        if self.disk_used_bytes() >= self.config.disk_cap_bytes * self.config.disk_soft_frac:
            self.rotate_results()
        # Brief breather so the next experiment doesn't immediately retrip.
        time.sleep(self.config.poll_interval_s)

    def rotate_results(self) -> None:
        """Gzip the oldest result files into the archive dir, then delete originals,
        until the results dir is below ``rotate_keep_bytes``."""
        files = sorted(
            (p for p in self.results_dir.glob("**/*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        archived = 0
        for fp in files:
            cur = sum(p.stat().st_size for p in self.results_dir.glob("**/*") if p.is_file())
            if cur <= self.config.rotate_keep_bytes:
                break
            rel = fp.relative_to(self.results_dir)
            arc = self.archive_dir / (str(rel) + ".gz")
            arc.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(fp, "rb") as src, gzip.open(arc, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
                fp.unlink()
                archived += 1
            except OSError as e:  # disk full, permissions, etc — abort rotation
                log.warning("rotate: %s -> %s failed: %s", fp, arc, e)
                break
        if archived:
            log.info("rotate: archived %d result files", archived)
        # If even the archive blew the cap, trim oldest archives outright.
        while self.disk_used_bytes() > self.config.disk_cap_bytes * self.config.disk_soft_frac:
            arc_files = sorted(
                (p for p in self.archive_dir.glob("**/*") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
            if not arc_files:
                break
            try:
                arc_files[0].unlink()
            except OSError:
                break

    def wait_until_clear(self, *, max_wait_s: float = 60.0) -> Pressure:
        """Block until pressure is NONE or HARD. Returns the terminal state."""
        deadline = time.monotonic() + max_wait_s
        while True:
            p = self.check()
            if p == Pressure.NONE or p == Pressure.HARD:
                return p
            self.relieve()
            if time.monotonic() >= deadline:
                return self.check()


__all__ = ["Governor", "GovernorConfig", "Pressure", "GIB"]
