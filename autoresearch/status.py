"""Quick status printer for the autoresearch loop. Run as a module:

    python -m autoresearch.status

Reports queue stats, recent results, and current resource use against caps.
Safe to run while the loop is active — read-only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .governor import GIB, Governor, GovernorConfig
from .loop import DEFAULT_STATE_DIR
from .queue import Queue


def main() -> int:
    state_dir = Path(os.environ.get("AUTORESEARCH_STATE_DIR", str(DEFAULT_STATE_DIR)))
    if not state_dir.is_dir():
        print(f"no state dir at {state_dir}; loop has not run yet.")
        return 1
    q = Queue(state_dir)
    g = Governor(state_dir, GovernorConfig(
        ram_cap_bytes=int(float(os.environ.get("AUTORESEARCH_RAM_GB", "30")) * GIB),
        disk_cap_bytes=int(float(os.environ.get("AUTORESEARCH_DISK_GB", "10")) * GIB),
    ))
    stats = q.stats()
    ram = g.ram_used_bytes()
    disk = g.disk_used_bytes()
    print(f"state_dir: {state_dir}")
    print(f"  pending: {stats['pending']}")
    print(f"  done:    {stats['done']}")
    print(f"  failed:  {stats['failed']}")
    print(f"  ram:     {ram / GIB:6.2f} GiB / {g.config.ram_cap_bytes / GIB:.1f}")
    print(f"  disk:    {disk / GIB:6.2f} GiB / {g.config.disk_cap_bytes / GIB:.1f}")
    # Last 5 done records:
    done = state_dir / "done.jsonl"
    if done.is_file():
        with open(done) as f:
            lines = f.readlines()[-5:]
        if lines:
            print("\nrecent done:")
            for line in lines:
                try:
                    r = json.loads(line)
                    res = r.get("result", {})
                    summary = {k: v for k, v in res.items() if k in ("status", "first_correct_step",
                                                                       "peak_kl_step", "first_would_halt_step",
                                                                       "duration_s")}
                    print(f"  {r['id']}: {summary}")
                except json.JSONDecodeError:
                    continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
