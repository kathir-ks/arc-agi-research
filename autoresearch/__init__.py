"""Phase 2 autoresearch loop.

A daemon that picks experiments off a persistent queue, runs them on CPU under
a strict RAM + disk budget, records results, and spawns followups so it never
runs out of work. Restart-safe: queue + done log live on disk.

Entrypoint: ``python -m autoresearch.loop`` (or ``scripts/run_autoresearch.sh``
for a restart-on-crash wrapper).
"""
