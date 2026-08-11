# DynamoRouter — Design Spec

**Date:** 2026-08-11
**Status:** Approved

## Summary

Rust inference request router fronting 8 Triton Inference Server replicas (mocked for development), with prefix-cache-aware scheduling: the router inspects incoming prompts, tracks which replica holds which KV-cache prefixes, and pins requests to the replica with the longest matching prefix. Goals: high throughput with low routing overhead (P99 < 8ms), large cache-hit-rate lift vs round-robin under ShareGPT trace replay, full OpenTelemetry + Prometheus instrumentation.

## Decisions

| Axis | Decision |
|------|----------|
| Backends | Mock Triton replicas speaking KServe gRPC protocol (router stays 100% Triton-compatible; can point at real Triton later) |
| Cache knowledge | Router-side prefix tracking (approximate per-replica radix tree built from routed traffic; no replica cooperation) |
| Client API | Triton KServe gRPC passthrough (`ModelInfer` + `ModelStreamInfer`) — drop-in proxy |
| Trace source | Public dataset replay (ShareGPT/LMSYS multi-turn conversations) |
| Deployment | docker-compose full stack: router, 8 mock replicas, Prometheus, Grafana, Jaeger |
| Matching/scheduling | Token radix tree (16-token blocks) + cost-based scheduler (approach A) |

## Architecture

Cargo workspace, 4 crates:

```
dynamorouter/
├── crates/
│   ├── router/        # main binary: gRPC proxy + scheduler
│   ├── mock-triton/   # mock replica binary, KServe gRPC protocol
│   ├── loadgen/       # ShareGPT trace replay + policy comparison
│   └── protocol/      # shared: generated KServe protos, common types
├── docker/            # Dockerfiles, compose, Grafana dashboards, Prometheus config
└── docs/
```

Data path:

```
client ──KServe gRPC──▶ router ──KServe gRPC──▶ mock-triton × 8
                          │
                          ├─ tokenizer (HF tokenizers, Llama)
                          ├─ scheduler: per-replica radix tree + load scores
                          ├─ /metrics (Prometheus)
                          └─ OTel traces ──▶ Jaeger
```

- Tokio multi-threaded runtime; `tonic` for gRPC on both sides.
- Persistent HTTP/2 connection per replica; requests multiplexed.
- Unary and streaming inference both proxied; streaming chunks forwarded as they arrive, zero buffering.
- Hot path: tokenize → radix lookup → score → route. Budget well under 8ms P99.

## Components

### Scheduler (router crate)

- `RadixTree` per replica. Nodes keyed on 16-token blocks: each block hashed to u64; tree over block-hash sequences. Insert on route decision; nodes carry last-access time.
- Capacity model: per-replica block budget (configurable, default 64K blocks). Over budget → evict LRU leaves. Mirrors the mock replica's own LRU so the router's picture stays honest.
- Scoring: `score(replica) = α · matched_blocks − β · inflight_requests`; route to argmax. Best match < 2 blocks → pure least-loaded fallback (avoids pinning cold traffic). α, β configurable.
- Concurrency: scheduler state behind a single `Mutex` (decision is microseconds). Measure contention; shard if needed.
- `--policy=round-robin` flag bypasses cache logic (baseline for benchmark).

### Mock replica (mock-triton crate)

- Implements KServe `GRPCInferenceService`: `ModelInfer`, `ModelStreamInfer`, `ServerLive`, `ServerReady`, `ModelMetadata`.
- Simulated KV cache: same 16-token block LRU, configurable capacity.
- On request: compute actual cached-prefix length → `prefill_delay = uncached_tokens × per_token_cost` (tokio sleep), then stream N decode chunks at fixed pace. Cache hits genuinely reduce latency.
- Trailing metadata reports cached/total block counts → ground-truth hit-rate measured at replica, not router's guess.

### Loadgen (loadgen crate)

- Parses ShareGPT JSON. Replays conversations multi-turn: turn k prompt = concatenation of turns 1..k (growing shared prefixes).
- Configurable concurrency, request rate, seed (deterministic).
- Runs both policies (cache-aware vs round-robin) for comparison.
- Outputs: throughput, P50/P99 latency, aggregate cache-hit rate from replica metadata.

### Observability

- Prometheus (router): `router_requests_total`, `router_routing_overhead_seconds` histogram, `router_cache_matched_blocks`, per-replica in-flight gauges. Replica: actual hit/miss counters.
- OTel: span per request; route decision + upstream call as child spans; OTLP export → Jaeger.
- Grafana: prebuilt dashboard JSON — cache hit rate, routing overhead P99, req/s, per-replica load balance.

## Error handling

- **Replica health:** connect failure or deadline → mark unhealthy, remove from scheduling, drop its radix tree (cache assumed lost). Background probe `ServerLive` every 2s → re-admit with empty tree.
- **Per-request:** upstream error → one retry on a different replica (failed one excluded), then propagate gRPC status. Mid-stream failure propagates to client; no replay.
- **Backpressure:** per-replica in-flight cap; all saturated → `RESOURCE_EXHAUSTED` immediately, no unbounded queueing.
- **Bad input:** tokenizer failure / oversized prompt → `INVALID_ARGUMENT`; router never panics on input.

## Testing

- **Unit:** radix tree insert/match/LRU-evict, scoring, block hashing — pure logic, heaviest coverage. TDD throughout.
- **Integration:** router + 2–3 mock replicas in-process (tokio test). Assert: same-prefix requests pin to same replica; failover works; round-robin flag bypasses cache logic; streaming passthrough intact.
- **Benchmark (not CI):** compose stack + loadgen ShareGPT replay, both policies, comparison table. Criterion micro-bench for route-decision latency.

## Success criteria

1. Cache-aware vs round-robin shows large hit-rate lift on ShareGPT replay (34%→72% class).
2. Routing decision overhead P99 < 8ms under high load.
3. `docker compose up` → live Grafana dashboard showing hit rate, P99 overhead, per-replica balance.
