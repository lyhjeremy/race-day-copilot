#!/bin/bash
# Auto-resuming wrapper around gen_scenarios.py. The generator self-stops
# cleanly after 3 consecutive claude -p failures (transient rate limiting)
# and is idempotent/resumable, so this loop just keeps re-invoking it with
# a backoff until the full 1454-scenario grid is done.
cd "$(dirname "$0")/.."
TARGET=1454
BACKOFF=600  # 10 min between retries after a self-stop

# Real bug found 2026-07-16: src/cache.py's embedder loads a fresh
# SentenceTransformer() on every process start, which does a network round-
# trip to check HF Hub for model updates even though all-MiniLM-L6-v2 is
# already fully cached locally. A network blip during this loop's unattended
# overnight run made every single restart fail identically and immediately
# with "RuntimeError: Cannot send a request, as the client has been closed"
# (an httpx/huggingface_hub client-reuse bug surfaced by the failed request),
# not the rate-limit self-stop this loop is designed to survive -- so it
# looped forever without making progress instead of backing off usefully.
# Forcing offline mode removes the network dependency for an asset that
# doesn't need it, rather than papering over the crash with more retries.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

while true; do
  COUNT=$(wc -l < data/scenarios.jsonl | tr -d ' ')
  if [ "$COUNT" -ge "$TARGET" ]; then
    echo "[loop] target reached: $COUNT/$TARGET"
    break
  fi
  echo "[loop] starting run: $COUNT/$TARGET done"
  python3 scripts/gen_scenarios.py
  COUNT=$(wc -l < data/scenarios.jsonl | tr -d ' ')
  if [ "$COUNT" -ge "$TARGET" ]; then
    echo "[loop] target reached: $COUNT/$TARGET"
    break
  fi
  echo "[loop] run exited, $COUNT/$TARGET done, backing off ${BACKOFF}s before retry"
  sleep "$BACKOFF"
done
echo "[loop] done"
