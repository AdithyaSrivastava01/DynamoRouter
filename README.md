# DynamoRouter

Prefix-cache-aware inference request router for Triton Inference Server, in Rust.

```
                          ┌────────────────────┐
                          │       client        │
                          └──────────┬───────────┘
                                     │ KServe gRPC (ModelInfer /
                                     │ ModelStreamInfer)
                                     ▼
                          ┌────────────────────┐
                          │       router        │
                          │  ┌──────────────┐  │
                          │  │  tokenizer    │  │
                          │  └──────┬───────┘  │
                          │         ▼            │
                          │  ┌──────────────┐  │   :9100/metrics (Prometheus)
                          │  │  scheduler    │──┼─▶ OTEL_EXPORTER_OTLP_ENDPOINT
                          │  │ (radix trees, │  │   (Jaeger)
                          │  │  per replica) │  │
                          │  └──────┬───────┘  │
                          └─────────┼────────────┘
                    ┌────────┬──────┼──────┬────────┐
                    ▼        ▼      ▼      ▼        ▼
                ┌───────┐┌───────┐┌───────┐    ┌───────┐
                │mock1  ││mock2  ││ ...   │    │mock8  │
                │:8001  ││:8001  ││       │    │:8001  │
                │KV sim ││KV sim ││       │    │KV sim │
                └───────┘└───────┘└───────┘    └───────┘
```

## How it works

The router speaks Triton's KServe gRPC protocol (`ModelInfer` / `ModelStreamInfer`)
on both sides: it looks like a Triton server to clients, and it forwards requests
to real (or, here, mocked) Triton replicas as a Triton client. Unary requests are
proxied with a bounded retry against a second replica on transport failure;
streaming requests are relayed chunk-by-chunk with zero buffering, so time-to-first-token
is untouched by the router.

To decide *where* to send a request, the router keeps an approximate picture of
each replica's KV cache: prompts are tokenized and folded into a chain of FNV-1a
hashes over 16-token blocks, where each block's hash also depends on every block
before it — so two requests only share a hash if they share the exact token
prefix, not just token content. Each replica gets its own radix tree over these
hashes; walking a tree with a new request's hash sequence counts how many leading
blocks that replica already has cached. The tree is a capacity-bounded LRU model
of the replica's real cache, with O(log n) lazy-heap eviction so it never grows
unbounded as traffic accumulates.

Routing itself is a per-request cost comparison: `score = α·matched_blocks −
β·inflight`, so the router favors the replica with the deepest cache hit unless
another replica is meaningfully less loaded. Requests whose best match falls
below a minimum-block threshold are treated as cold traffic and sent to whichever
healthy replica has the fewest in-flight requests, rather than pinning cold
prompts to a single replica. A background health prober periodically checks
`ServerLive` on every replica, dropping a replica's radix tree (its cache is
assumed lost) when it goes unhealthy and re-admitting it clean when it recovers.

## Results (ShareGPT replay, 8 mock replicas)

Measured with `CONVS=500 CONC=64 ./scripts/bench.sh` against the full docker
compose stack (8 mock Triton replicas, default `CACHE_BLOCKS=4096` per replica,
`PREFILL_US_PER_TOKEN=200`), replaying 500 multi-turn ShareGPT conversations
(1643 requests) at concurrency 64:

| policy | cache hit rate | P50 | P99 | throughput |
|--------|----------------|-----|------|------------|
| round-robin | 14.5% (13496/92977 blocks) | 143.1 ms | 511.1 ms | 277 req/s |
| cache-aware | 59.8% (55631/92977 blocks) | 73.0 ms | 390.6 ms | 648 req/s |

Cache-aware scheduling more than quadruples the ground-truth cache hit rate over
round-robin, roughly halves P50 latency, and more than doubles throughput — all
from the same replica fleet, since round-robin scatters a conversation's
follow-up turns across replicas that never saw its prefix while cache-aware
keeps a conversation pinned to the replica that already holds it.

## Quickstart

```bash
# One-time data fetch (GPT-2 tokenizer.json + ShareGPT trace, ~640MB)
./scripts/fetch_tokenizer.sh data
./scripts/fetch_sharegpt.sh data

# Bring up router + 8 mock replicas + Prometheus + Grafana + Jaeger
docker compose -f docker/docker-compose.yml up --build -d

# Run the benchmark (round-robin, then cache-aware)
CONVS=500 CONC=64 ./scripts/bench.sh
```

- Grafana: http://localhost:3000 (anonymous, admin role) — "DynamoRouter" dashboard
- Jaeger: http://localhost:16686 — per-request `route_decision` / `upstream_infer` spans
- Prometheus: http://localhost:9090

Tear down with `docker compose -f docker/docker-compose.yml down`.

## Design docs

- `docs/superpowers/specs/2026-08-11-dynamorouter-design.md`
- `docs/superpowers/plans/2026-08-11-dynamorouter.md`
