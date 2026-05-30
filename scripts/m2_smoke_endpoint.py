#!/usr/bin/env python3
"""Endpoint smoke test for MiniMax-M2.7 on main-2.

Sends N short completion requests serially and reports pass/fail per request.
Used as the v1 acceptance gate for "endpoint serves N completions without
OOM/hang." Defaults to 10 requests, ~30 tokens each.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


PROMPTS = [
    "What is the capital of France?",
    "List three primes between 10 and 30.",
    "Translate to French: 'Good morning, how are you?'",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain backpropagation in one sentence.",
    "What is 17 * 23?",
    "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    "Write one haiku about recursion.",
    "Give the SI unit of electrical resistance.",
    "Reverse the string 'palindrome'.",
]


def call_once(endpoint: str, model: str, prompt: str, max_tokens: int,
              timeout: float) -> dict:
  body = json.dumps({
      "model": model,
      "prompt": prompt,
      "max_tokens": max_tokens,
      "temperature": 0.0,
  }).encode("utf-8")
  req = urllib.request.Request(
      f"{endpoint}/v1/completions",
      data=body,
      headers={"Content-Type": "application/json"},
  )
  t0 = time.perf_counter()
  with urllib.request.urlopen(req, timeout=timeout) as r:
    payload = json.load(r)
  dt = time.perf_counter() - t0
  text = (payload.get("choices") or [{}])[0].get("text", "")
  usage = payload.get("usage") or {}
  return {
      "wall_seconds": round(dt, 2),
      "completion_tokens": usage.get("completion_tokens", 0),
      "text_preview": text[:80],
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--endpoint",
                 default="http://main-2.us-central2-b.c.optical-pillar-442815-p2.internal:8000")
  p.add_argument("--model", default="minimax-m2.7")
  p.add_argument("--n-prompts", type=int, default=10)
  p.add_argument("--max-tokens", type=int, default=32)
  p.add_argument("--timeout", type=float, default=300.0)
  args = p.parse_args()

  prompts = (PROMPTS * ((args.n_prompts + len(PROMPTS) - 1) // len(PROMPTS)))[:args.n_prompts]

  results, fails = [], 0
  total_t0 = time.perf_counter()
  for i, prompt in enumerate(prompts):
    try:
      r = call_once(args.endpoint, args.model, prompt, args.max_tokens, args.timeout)
      r["status"] = "ok"
      r["prompt"] = prompt
      results.append(r)
      print(f"[{i+1:02d}/{len(prompts)}] OK {r['wall_seconds']}s "
            f"({r['completion_tokens']} tok): {r['text_preview']!r}")
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
      fails += 1
      results.append({"status": "fail", "prompt": prompt, "error": str(e)})
      print(f"[{i+1:02d}/{len(prompts)}] FAIL: {e}")

  wall = time.perf_counter() - total_t0
  print("\n=== summary ===")
  print(json.dumps({
      "endpoint": args.endpoint,
      "prompts_sent": len(prompts),
      "prompts_ok": len(prompts) - fails,
      "prompts_failed": fails,
      "wall_seconds_total": round(wall, 2),
      "verdict": "PASS" if fails == 0 else "FAIL",
  }, indent=2))
  raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
  main()
