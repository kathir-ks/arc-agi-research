#!/usr/bin/env bash
# Restart-on-crash wrapper for the autoresearch loop.
#
# Exit codes:
#   0  – clean exit (budget elapsed or signal). Wrapper stops.
#   42 – governor requested restart (RAM/disk pressure or runaway failures).
#        Wrapper restarts after backoff.
#   *  – anything else (segfault, OOM-killed, uncaught exception).
#        Wrapper restarts with exponential backoff up to MAX_BACKOFF.
#
# Usage:
#   ./scripts/run_autoresearch.sh                  # foreground forever
#   nohup ./scripts/run_autoresearch.sh &>/tmp/autoresearch.log &  # detached

set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

LOG_FILE="${AUTORESEARCH_WRAPPER_LOG:-$ROOT/autoresearch/state/wrapper.log}"
mkdir -p "$(dirname "$LOG_FILE")"

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') wrapper: $*" | tee -a "$LOG_FILE"; }

BACKOFF=5
MAX_BACKOFF=300

trap 'log "wrapper SIGTERM/SIGINT; exiting"; exit 0' SIGTERM SIGINT

while true; do
  log "starting autoresearch loop (pid will follow)"
  /usr/bin/python3 -m autoresearch.loop
  CODE=$?
  case "$CODE" in
    0)
      log "loop exited cleanly (code 0); wrapper stopping"
      exit 0
      ;;
    42)
      log "loop requested restart (code 42); backoff=${BACKOFF}s"
      sleep "$BACKOFF"
      BACKOFF=5  # restart-by-request: don't escalate backoff
      ;;
    *)
      log "loop crashed (code $CODE); backoff=${BACKOFF}s"
      sleep "$BACKOFF"
      BACKOFF=$(( BACKOFF * 2 ))
      if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then BACKOFF=$MAX_BACKOFF; fi
      ;;
  esac
done
