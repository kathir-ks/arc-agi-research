"""Sonnet 4.6 subprocess wrapper for LLM-driven experiment proposals.

Invokes ``claude -p`` non-interactively with these guards:

  * ``--model claude-sonnet-4-6``           – the model the user requested
  * ``--max-budget-usd 0.05`` (default)    – hard per-call dollar cap
  * ``--permission-mode bypassPermissions``– no interactive prompts (overridable
                                              via env for in-host smoke tests)
  * ``--disallowedTools "Bash,Edit,..."``  – explicit deny-list of every tool
                                              the subagent could use. With this
                                              the subprocess can only produce
                                              text — no filesystem, no shell,
                                              no web. This is the actual
                                              sandboxing.
  * ``--disable-slash-commands``           – no skills, no surprises
  * ``--no-session-persistence``           – no resumable session, no on-disk
                                              session record
  * ``--output-format json``                – wraps the LLM's reply in a
                                              machine-readable envelope with
                                              cost / error fields.

We deliberately do NOT pass ``--json-schema`` — empirically that mode burns the
entire $0.05 budget on cache-creation tokens before any output is generated.
We instruct the LLM to reply with strict JSON in the prompt and strip any
`````json`` fences in :func:`_extract_json` defensively.

The wrapper never throws on a bad LLM response — it returns a structured error
dict the loop can record. The autoresearch daemon must keep running even when
the proposer is unavailable.

Env overrides:
  AUTORESEARCH_PROPOSER_MODEL      (default: claude-sonnet-4-6)
  AUTORESEARCH_PROPOSER_BUDGET_USD (default: 0.05)
  AUTORESEARCH_PROPOSER_TIMEOUT_S  (default: 120)
  AUTORESEARCH_PROPOSER_PERMISSION (default: bypassPermissions)
  AUTORESEARCH_PROPOSER_DISALLOWED (default: a broad deny-list, below)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("autoresearch.claude_proposer")

DEFAULT_MODEL = "claude-sonnet-4-6"
# Per-call hard cap. Empirically observed costs per call:
#   * cold cache, short prompt:  ~$0.06
#   * cold cache, fuller prompt: ~$0.10–0.11
#   * warm cache (within 5-min TTL): ~$0.02
# We default to $0.15 to leave headroom; AUTORESEARCH_PROPOSER_BUDGET_USD
# overrides. When the cap is hit mid-call, claude reports
# subtype=error_max_budget_usd and we surface "budget exhausted" upstream.
DEFAULT_BUDGET_USD = 0.15
DEFAULT_TIMEOUT_S = 180

# Every tool we can think of, comma-separated. The subagent has access to NONE
# of these and so can only generate text. Comma-form is required — space-form
# is parsed by claude's CLI as multiple positional args and eats the prompt.
DEFAULT_DISALLOWED_TOOLS = (
    "Bash,Edit,Write,Read,Glob,Grep,Task,Agent,WebFetch,WebSearch,NotebookEdit,"
    "BashOutput,KillShell,SlashCommand,Skill,ToolSearch,SendMessage,SendUserFile,"
    "TaskCreate,TaskUpdate,TaskList,TaskGet,TaskOutput,TaskStop,Monitor,"
    "PushNotification,RemoteTrigger,CronCreate,CronList,CronDelete,"
    "EnterPlanMode,ExitPlanMode,EnterWorktree,ExitWorktree,ScheduleWakeup,"
    "AskUserQuestion"
)


@dataclass
class ProposerResult:
    ok: bool
    analysis: str
    proposals: list[dict[str, Any]]
    cost_usd: float
    duration_s: float
    error: str | None = None
    raw: dict[str, Any] | None = None


def _build_argv(prompt: str) -> list[str]:
    model = os.environ.get("AUTORESEARCH_PROPOSER_MODEL", DEFAULT_MODEL)
    budget = os.environ.get("AUTORESEARCH_PROPOSER_BUDGET_USD", str(DEFAULT_BUDGET_USD))
    perm = os.environ.get("AUTORESEARCH_PROPOSER_PERMISSION", "bypassPermissions")
    disallowed = os.environ.get("AUTORESEARCH_PROPOSER_DISALLOWED", DEFAULT_DISALLOWED_TOOLS)
    # NOTE: -p comes LAST (before the prompt) because --disallowedTools takes a
    # space-separated list with nargs > 0 — placing -p after --disallowedTools
    # disambiguates the argument boundary.
    return [
        "claude",
        "--model", model,
        "--output-format", "json",
        "--max-budget-usd", budget,
        "--permission-mode", perm,
        "--disallowedTools", disallowed,
        "--disable-slash-commands",
        "--no-session-persistence",
        "-p",
        prompt,
    ]


_FENCE_PREFIXES = ("```json", "```JSON", "```")


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a possibly markdown-wrapped LLM reply."""
    s = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    for prefix in _FENCE_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip("\n")
            if s.endswith("```"):
                s = s[: -len("```")].rstrip()
            break
    # Try direct parse first.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first { ... } block.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_prompt(
    recent_done: list[dict[str, Any]],
    *,
    registered_types: list[str],
    canonical_sample: list[str],  # kept for signature compatibility; unused
) -> str:
    """Compose a compact, self-contained prompt. Trimmed to keep token count
    (and per-call cost) bounded — we only show the 12 most recent results,
    and only the fields a proposer needs."""
    trimmed: list[dict[str, Any]] = []
    for r in recent_done[-12:]:
        res = r.get("result") or {}
        # Skip skipped / failed runs; they're noise for the proposer.
        if res.get("status") != "ok":
            continue
        trimmed.append(
            {
                "t": r.get("type", "").replace("exp_", ""),
                "stem": (r.get("params") or {}).get("stem"),
                "fc": res.get("first_correct_step"),     # exp_a
                "kls": res.get("peak_kl_step"),           # exp_c
                "klv": res.get("peak_kl_value"),
                "halt": res.get("first_would_halt_step"), # exp_d
                "slow": res.get("slow_convergence"),
            }
        )

    return (
        "You are inside a TRM (Tiny Recursive Model) interpretability loop. "
        f"Registered experiment types: {registered_types}. "
        "exp_a_convergence = per-iter argmax/entropy; "
        "exp_c_ablate = task-id KL ablation; "
        "exp_d_halt = q_halt trajectory. "
        "Stems are 8-char ARC task IDs (e.g. '007bbfb7'). "
        "fc=first_correct_step (-1 = never), kls=peak_kl_step, klv=peak_kl_value, halt=first_would_halt_step (-1 = never). "
        "Recent results:\n"
        + json.dumps(trimmed, separators=(",", ":"))
        + "\n\nPropose 2-5 followup experiments to investigate the most interesting tasks above. "
        'Reply with ONLY this JSON (no markdown, no prose): '
        '{"analysis": "1-2 sentences", "proposals": [{"type": "<one of registered>", '
        '"stem": "<from results above>", "rationale": "1 line why"}]}'
    )


def run_proposer(prompt: str, *, timeout_s: int | None = None) -> ProposerResult:
    timeout = timeout_s or int(float(os.environ.get("AUTORESEARCH_PROPOSER_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    argv = _build_argv(prompt)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=0.0,
            duration_s=time.monotonic() - t0,
            error=f"timeout after {timeout}s",
        )
    duration = time.monotonic() - t0

    if not proc.stdout.strip():
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=0.0, duration_s=duration,
            error=f"empty stdout (rc={proc.returncode}); stderr={proc.stderr[:300]!r}",
        )

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=0.0, duration_s=duration,
            error=f"non-JSON envelope: {e}; first 200 bytes={proc.stdout[:200]!r}",
        )

    cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)
    if envelope.get("is_error"):
        # Surface the specific failure subtype so the loop can log it usefully.
        subtype = envelope.get("subtype", "")
        errs = envelope.get("errors") or []
        if subtype == "error_max_budget_usd" or any("budget" in str(e).lower() for e in errs):
            detail = f"max_budget_usd exhausted at ${cost:.4f}"
        else:
            raw_result_dbg = envelope.get("result")
            head = "" if raw_result_dbg is None else str(raw_result_dbg)[:200]
            detail = f"subtype={subtype!r} errors={errs!r} result_head={head!r}"
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=cost, duration_s=duration,
            error=detail,
            raw=envelope,
        )

    raw_result = envelope.get("result")
    # The "result" field is the LLM's text response. We instruct it to reply with
    # JSON only, but the LLM often wraps in ```json fences anyway.
    if isinstance(raw_result, str):
        payload = _extract_json(raw_result)
        if payload is None:
            # Try parsing as a top-level JSON array too.
            try:
                payload = json.loads(raw_result.strip().strip("`").lstrip("json").strip())
            except (json.JSONDecodeError, ValueError):
                return ProposerResult(
                    ok=False, analysis="", proposals=[], cost_usd=cost, duration_s=duration,
                    error=f"result text not parseable as JSON; raw={raw_result[:200]!r}",
                    raw=envelope,
                )
    elif isinstance(raw_result, (dict, list)):
        payload = raw_result
    else:
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=cost, duration_s=duration,
            error=f"unexpected result type: {type(raw_result).__name__}",
            raw=envelope,
        )

    # Accept either {analysis, proposals: [...]}  or  a bare proposals list.
    if isinstance(payload, list):
        analysis = ""
        raw_proposals: list[Any] = payload
    elif isinstance(payload, dict):
        analysis = str(payload.get("analysis", ""))
        raw_proposals = payload.get("proposals") or []
        if not isinstance(raw_proposals, list):
            return ProposerResult(
                ok=False, analysis=analysis, proposals=[], cost_usd=cost, duration_s=duration,
                error=f"proposals not a list: {type(raw_proposals).__name__}",
                raw=envelope,
            )
    else:
        return ProposerResult(
            ok=False, analysis="", proposals=[], cost_usd=cost, duration_s=duration,
            error=f"payload neither dict nor list: {type(payload).__name__}",
            raw=envelope,
        )
    proposals = [p for p in raw_proposals if isinstance(p, dict) and "type" in p and "stem" in p]

    return ProposerResult(
        ok=True, analysis=analysis, proposals=proposals,
        cost_usd=cost, duration_s=duration, raw=envelope,
    )


__all__ = ["ProposerResult", "DEFAULT_DISALLOWED_TOOLS", "build_prompt", "run_proposer"]
