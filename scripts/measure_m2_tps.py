#!/usr/bin/env python3
"""Throughput sanity check for the MiniMax-M2.7 endpoint on main-2.

Sends one `/v1/completions` request, measures wall clock per output token,
prints `decode_tokens_per_second` for the acceptance log.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--endpoint",
                 default="http://main-2.us-central2-b.c.optical-pillar-442815-p2.internal:8000",
                 help="Base URL of the maxtext_server")
  p.add_argument("--model", default="minimax-m2.7")
  p.add_argument("--prompt",
                 default="Write a short story about a recursive transformer.",
                 help="Decode prompt — keep short so prefill isn't a confound")
  p.add_argument("--n-tokens", type=int, default=512,
                 help="max_tokens to request")
  p.add_argument("--temperature", type=float, default=0.0)
  args = p.parse_args()

  body = json.dumps({
      "model": args.model,
      "prompt": args.prompt,
      "max_tokens": args.n_tokens,
      "temperature": args.temperature,
      "stream": False,
  }).encode("utf-8")

  req = urllib.request.Request(
      f"{args.endpoint}/v1/completions",
      data=body,
      headers={"Content-Type": "application/json"},
  )

  t0 = time.perf_counter()
  try:
    with urllib.request.urlopen(req, timeout=600) as r:
      payload = json.load(r)
  except urllib.error.URLError as e:
    raise SystemExit(f"endpoint unreachable: {e}")
  dt = time.perf_counter() - t0

  usage = payload.get("usage") or {}
  prompt_tokens = usage.get("prompt_tokens", 0)
  completion_tokens = usage.get("completion_tokens", 0)
  total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

  if completion_tokens == 0:
    raise SystemExit("server returned 0 completion tokens — something is wrong")

  decode_tps = completion_tokens / dt
  print(json.dumps({
      "endpoint": args.endpoint,
      "wall_seconds": round(dt, 3),
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "total_tokens": total_tokens,
      "decode_tokens_per_second": round(decode_tps, 2),
      "first_64_chars": (payload.get("choices") or [{}])[0].get("text", "")[:64],
  }, indent=2))


if __name__ == "__main__":
  main()
