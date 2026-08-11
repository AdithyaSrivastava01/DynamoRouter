#!/usr/bin/env bash
# Compare cache-aware vs round-robin on ShareGPT replay against the compose stack.
set -euo pipefail
cd "$(dirname "$0")/.."

CONVS="${CONVS:-500}"
CONC="${CONC:-64}"
TRACE="${TRACE:-data/sharegpt.json}"

./scripts/fetch_tokenizer.sh data
./scripts/fetch_sharegpt.sh data
cargo build --release -p loadgen

run_policy() {
  local policy="$1"
  echo "--- restarting router with POLICY=$policy ---"
  POLICY="$policy" docker compose -f docker/docker-compose.yml up -d --force-recreate router
  sleep 3
  ./target/release/loadgen \
    --router http://127.0.0.1:8000 \
    --trace "$TRACE" \
    --conversations "$CONVS" \
    --concurrency "$CONC" \
    --label "$policy"
}

run_policy round-robin
run_policy cache-aware
