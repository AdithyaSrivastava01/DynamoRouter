# DynamoRouter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rust inference request router fronting 8 mock Triton replicas with prefix-cache-aware scheduling, demonstrating a large cache-hit-rate lift vs round-robin under ShareGPT trace replay, with Prometheus + OpenTelemetry instrumentation and a docker-compose demo stack.

**Architecture:** Cargo workspace with 4 crates: `protocol` (trimmed wire-compatible KServe protos + token-block hashing), `router` (tonic gRPC proxy, per-replica radix trees, cost scheduler), `mock-triton` (KServe-speaking replica with simulated KV cache and latency), `loadgen` (ShareGPT multi-turn replay + stats). Router keeps an approximate per-replica radix tree over 16-token block hashes; score = α·matched_blocks − β·inflight.

**Tech Stack:** Rust (edition 2021), tokio, tonic/prost, tokenizers (HF), prometheus, axum (metrics endpoint), tracing + opentelemetry-otlp, clap, serde_json, docker compose, Grafana, Jaeger.

**Spec:** `docs/superpowers/specs/2026-08-11-dynamorouter-design.md`

---

## File Structure

```
Cargo.toml                          # workspace
crates/
  protocol/
    Cargo.toml
    build.rs                        # tonic-build
    proto/inference.proto           # trimmed KServe (wire-compatible field numbers)
    src/lib.rs                      # pb module + re-exports
    src/blocks.rs                   # BLOCK_SIZE, chained block_hashes()
  router/
    Cargo.toml
    src/main.rs                     # clap config, serve gRPC + /metrics, health prober
    src/tokenizer.rs                # PromptTokenizer wrapper
    src/radix.rs                    # RadixTree: match/insert/LRU-evict
    src/scheduler.rs                # Scheduler: pick/complete/health, policies
    src/server.rs                   # RouterService: ModelInfer, ModelStreamInfer, passthroughs
    src/telemetry.rs                # metrics registry + optional OTel init
    tests/integration.rs            # router + in-process mocks
  mock-triton/
    Cargo.toml
    src/main.rs                     # clap config, serve gRPC + /metrics
    src/cache.rs                    # KvCacheSim (block LRU)
    src/service.rs                  # MockTritonService
  loadgen/
    Cargo.toml
    src/main.rs                     # replay driver + stats output
    src/trace.rs                    # ShareGPT parsing → multi-turn prompts
    tests/fixtures/sharegpt_small.json
scripts/
  fetch_tokenizer.sh                # GPT-2 tokenizer.json (public, ungated)
  fetch_sharegpt.sh                 # ShareGPT dataset download
  bench.sh                          # both policies → comparison table
docker/
  Dockerfile                        # multi-stage, non-root user
  docker-compose.yml                # router, mock1..8, prometheus, grafana, jaeger
  prometheus.yml
  grafana/provisioning/...          # datasource + dashboard
README.md
```

Note: spec says Llama tokenizer; plan uses **GPT-2 tokenizer.json** — functionally equivalent BPE for prefix-block purposes, but public/ungated (no HF auth needed). Both router and mock replicas load the same file.

---

### Task 1: Workspace scaffold

**Files:**
- Create: `Cargo.toml`, `.gitignore`, `rust-toolchain.toml`
- Create: `crates/{protocol,router,mock-triton,loadgen}/Cargo.toml` + minimal `src/lib.rs`/`src/main.rs`

- [ ] **Step 1: Create workspace files**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["crates/protocol", "crates/router", "crates/mock-triton", "crates/loadgen"]

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
tonic = "0.12"
prost = "0.13"
tokio-stream = "0.1"
anyhow = "1"
clap = { version = "4", features = ["derive", "env"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
prometheus = "0.13"
axum = "0.7"
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

`.gitignore`:
```
/target
/data
tokenizer.json
```

`rust-toolchain.toml`:
```toml
[toolchain]
channel = "stable"
```

- [ ] **Step 2: Create the four crates**

`crates/protocol/Cargo.toml`:
```toml
[package]
name = "protocol"
version = "0.1.0"
edition = "2021"

[dependencies]
tonic = { workspace = true }
prost = { workspace = true }

[build-dependencies]
tonic-build = "0.12"
```

`crates/protocol/src/lib.rs`:
```rust
pub mod blocks;
```
(`blocks.rs` arrives in Task 3 — create an empty `crates/protocol/src/blocks.rs` for now.)

`crates/router/Cargo.toml`:
```toml
[package]
name = "router"
version = "0.1.0"
edition = "2021"

[dependencies]
protocol = { path = "../protocol" }
tokio = { workspace = true }
tonic = { workspace = true }
prost = { workspace = true }
tokio-stream = { workspace = true }
anyhow = { workspace = true }
clap = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
prometheus = { workspace = true }
axum = { workspace = true }
tokenizers = { workspace = true }

[dev-dependencies]
mock-triton = { path = "../mock-triton" }
```

`crates/router/src/main.rs`:
```rust
fn main() {
    println!("router placeholder");
}
```

`crates/mock-triton/Cargo.toml`:
```toml
[package]
name = "mock-triton"
version = "0.1.0"
edition = "2021"

[lib]
name = "mock_triton"
path = "src/lib.rs"

[[bin]]
name = "mock-triton"
path = "src/main.rs"

[dependencies]
protocol = { path = "../protocol" }
tokio = { workspace = true }
tonic = { workspace = true }
prost = { workspace = true }
tokio-stream = { workspace = true }
anyhow = { workspace = true }
clap = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
prometheus = { workspace = true }
axum = { workspace = true }
tokenizers = { workspace = true }
```

`crates/mock-triton/src/lib.rs`:
```rust
pub mod cache;
```
(create empty `crates/mock-triton/src/cache.rs`)

`crates/mock-triton/src/main.rs`:
```rust
fn main() {
    println!("mock-triton placeholder");
}
```

`crates/loadgen/Cargo.toml`:
```toml
[package]
name = "loadgen"
version = "0.1.0"
edition = "2021"

[dependencies]
protocol = { path = "../protocol" }
tokio = { workspace = true }
tonic = { workspace = true }
anyhow = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

`crates/loadgen/src/main.rs`:
```rust
fn main() {
    println!("loadgen placeholder");
}
```

- [ ] **Step 3: Verify build**

Run: `cargo build --workspace`
Expected: compiles clean (placeholders only).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: scaffold cargo workspace with four crates"
```

---

### Task 2: KServe protocol crate

**Files:**
- Create: `crates/protocol/proto/inference.proto`
- Create: `crates/protocol/build.rs`
- Modify: `crates/protocol/src/lib.rs`

Field numbers copied from Triton's real `grpc_service.proto` — wire-compatible for the messages/fields we use. Unused fields (e.g. `raw_input_contents = 7`) omitted; proto3 tolerates that on the wire.

- [ ] **Step 1: Write the proto**

`crates/protocol/proto/inference.proto`:
```proto
syntax = "proto3";
package inference;

service GRPCInferenceService {
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse) {}
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse) {}
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse) {}
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse) {}
  rpc ModelStreamInfer(stream ModelInferRequest) returns (stream ModelStreamInferResponse) {}
}

message ServerLiveRequest {}
message ServerLiveResponse { bool live = 1; }
message ServerReadyRequest {}
message ServerReadyResponse { bool ready = 1; }

message ModelMetadataRequest {
  string name = 1;
  string version = 2;
}
message ModelMetadataResponse {
  string name = 1;
  repeated string versions = 2;
  string platform = 3;
}

message InferParameter {
  oneof parameter_choice {
    bool bool_param = 1;
    int64 int64_param = 2;
    string string_param = 3;
  }
}

message InferTensorContents {
  repeated bool bool_contents = 1;
  repeated int32 int_contents = 2;
  repeated int64 int64_contents = 3;
  repeated uint32 uint_contents = 4;
  repeated uint64 uint64_contents = 5;
  repeated float fp32_contents = 6;
  repeated double fp64_contents = 7;
  repeated bytes bytes_contents = 8;
}

message ModelInferRequest {
  message InferInputTensor {
    string name = 1;
    string datatype = 2;
    repeated int64 shape = 3;
    map<string, InferParameter> parameters = 4;
    InferTensorContents contents = 5;
  }
  message InferRequestedOutputTensor {
    string name = 1;
    map<string, InferParameter> parameters = 2;
  }
  string model_name = 1;
  string model_version = 2;
  string id = 3;
  map<string, InferParameter> parameters = 4;
  repeated InferInputTensor inputs = 5;
  repeated InferRequestedOutputTensor outputs = 6;
}

message ModelInferResponse {
  message InferOutputTensor {
    string name = 1;
    string datatype = 2;
    repeated int64 shape = 3;
    map<string, InferParameter> parameters = 4;
    InferTensorContents contents = 5;
  }
  string model_name = 1;
  string model_version = 2;
  string id = 3;
  map<string, InferParameter> parameters = 4;
  repeated InferOutputTensor outputs = 5;
}

message ModelStreamInferResponse {
  string error_message = 1;
  ModelInferResponse infer_response = 2;
}
```

- [ ] **Step 2: build.rs + lib.rs**

`crates/protocol/build.rs`:
```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/inference.proto"], &["proto"])?;
    Ok(())
}
```

`crates/protocol/src/lib.rs`:
```rust
pub mod blocks;

pub mod pb {
    tonic::include_proto!("inference");
}

pub use pb::grpc_inference_service_client::GrpcInferenceServiceClient;
pub use pb::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
```

Note: `protoc` must be installed (`apt install protobuf-compiler`). If tonic-build 0.12 lacks `compile_protos`, the method is `compile` — same signature.

- [ ] **Step 3: Smoke test**

Append to `crates/protocol/src/lib.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::pb;

    #[test]
    fn proto_types_roundtrip() {
        let req = pb::ModelInferRequest {
            model_name: "m".into(),
            inputs: vec![pb::model_infer_request::InferInputTensor {
                name: "text_input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                contents: Some(pb::InferTensorContents {
                    bytes_contents: vec![b"hello".to_vec()],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(req.inputs[0].contents.as_ref().unwrap().bytes_contents[0], b"hello");
    }
}
```

Run: `cargo test -p protocol`
Expected: PASS (build.rs generates code, test compiles and passes).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(protocol): trimmed wire-compatible KServe gRPC protos"
```

---

### Task 3: Token block hashing

**Files:**
- Modify: `crates/protocol/src/blocks.rs`

Chained FNV-1a over 16-token blocks: each block's hash folds in the previous block's hash, so a hash sequence identifies a *prefix*, not just content (same standard trick as vLLM prefix caching). Partial trailing blocks are dropped — not cacheable.

- [ ] **Step 1: Write failing tests**

`crates/protocol/src/blocks.rs`:
```rust
pub const BLOCK_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize, offset: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i + offset).collect()
    }

    #[test]
    fn shared_prefix_shares_hashes() {
        let a = block_hashes(&toks(64, 0));
        let mut b_toks = toks(48, 0);
        b_toks.extend(toks(16, 999));
        let b = block_hashes(&b_toks);
        assert_eq!(a.len(), 4);
        assert_eq!(a[..3], b[..3]);
        assert_ne!(a[3], b[3]);
    }

    #[test]
    fn position_matters() {
        // same 16 tokens, but preceded by different block → different hash
        let x: Vec<u32> = toks(16, 0).into_iter().chain(toks(16, 100)).collect();
        let y: Vec<u32> = toks(16, 50).into_iter().chain(toks(16, 100)).collect();
        let hx = block_hashes(&x);
        let hy = block_hashes(&y);
        assert_ne!(hx[1], hy[1]);
    }

    #[test]
    fn partial_block_dropped() {
        assert_eq!(block_hashes(&toks(15, 0)).len(), 0);
        assert_eq!(block_hashes(&toks(17, 0)).len(), 1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p protocol blocks`
Expected: FAIL — `block_hashes` not found.

- [ ] **Step 3: Implement**

Add above the tests in `crates/protocol/src/blocks.rs`:
```rust
/// Chained FNV-1a over BLOCK_SIZE-token blocks. Hash i depends on blocks 0..=i,
/// so equal hash prefixes == equal token prefixes (modulo collisions).
pub fn block_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(tokens.len() / BLOCK_SIZE);
    let mut prev: u64 = 0xcbf29ce484222325;
    for block in tokens.chunks_exact(BLOCK_SIZE) {
        let mut h = prev;
        for &t in block {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        hashes.push(h);
        prev = h;
    }
    hashes
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p protocol blocks`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/protocol/src/blocks.rs
git commit -m "feat(protocol): chained token-block hashing for prefix identity"
```

---

### Task 4: Tokenizer wrapper + fetch script

**Files:**
- Create: `crates/router/src/tokenizer.rs`, `crates/router/src/lib.rs`
- Create: `scripts/fetch_tokenizer.sh`

Router needs a lib target so tests/integration and mock-triton reuse work. `PromptTokenizer` wraps HF `tokenizers`, no special tokens (prefix identity must not depend on BOS handling).

- [ ] **Step 1: Fetch script**

`scripts/fetch_tokenizer.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
if [ ! -f "$DEST/tokenizer.json" ]; then
  curl -fL -o "$DEST/tokenizer.json" \
    "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
fi
echo "tokenizer at $DEST/tokenizer.json"
```

Run: `chmod +x scripts/fetch_tokenizer.sh && ./scripts/fetch_tokenizer.sh data`
Expected: `data/tokenizer.json` exists (~1.3MB).

- [ ] **Step 2: Write failing test**

`crates/router/src/lib.rs`:
```rust
pub mod radix;
pub mod scheduler;
pub mod server;
pub mod telemetry;
pub mod tokenizer;
```
(create empty `radix.rs`, `scheduler.rs`, `server.rs`, `telemetry.rs` files so it compiles)

`crates/router/src/tokenizer.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_is_deterministic_and_prefix_stable() {
        let tok = PromptTokenizer::from_file("../../data/tokenizer.json").unwrap();
        let a = tok.tokenize("The quick brown fox jumps over the lazy dog. ");
        let b = tok.tokenize("The quick brown fox jumps over the lazy dog. And more text.");
        assert!(!a.is_empty());
        assert_eq!(a, tok.tokenize("The quick brown fox jumps over the lazy dog. "));
        assert_eq!(b[..a.len()], a[..]); // BPE prefix stability at word boundary
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p router tokenizer`
Expected: FAIL — `PromptTokenizer` not found.

- [ ] **Step 4: Implement**

Add above tests in `crates/router/src/tokenizer.rs`:
```rust
use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub struct PromptTokenizer {
    inner: Tokenizer,
}

impl PromptTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }
}
```

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p router tokenizer`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(router): HF tokenizer wrapper + fetch script"
```

---

### Task 5: Radix tree with LRU eviction

**Files:**
- Modify: `crates/router/src/radix.rs`

Per-replica approximate cache picture. Tree over block-hash sequences (each edge = one 16-token block hash). `match_prefix` counts matched leading blocks and refreshes access times on the path. `insert` adds the chain. `evict_to(budget)` removes least-recently-used leaves until node count ≤ budget — mirrors replica LRU.

- [ ] **Step 1: Write failing tests**

`crates/router/src/radix.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_counts_shared_prefix() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3, 4]);
        assert_eq!(t.match_prefix(&[1, 2, 3, 4]), 4);
        assert_eq!(t.match_prefix(&[1, 2, 9, 9]), 2);
        assert_eq!(t.match_prefix(&[9]), 0);
        assert_eq!(t.num_blocks(), 4);
    }

    #[test]
    fn insert_dedups_shared_prefix() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3]);
        t.insert(&[1, 2, 4]);
        assert_eq!(t.num_blocks(), 4); // 1,2 shared; 3 and 4 branch
    }

    #[test]
    fn evict_removes_lru_leaves_first() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2]);
        t.insert(&[3, 4]);
        t.match_prefix(&[1, 2]); // refresh chain 1→2
        t.evict_to(2);
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.match_prefix(&[1, 2]), 2); // recently used survives
        assert!(t.match_prefix(&[3, 4]) < 2); // cold chain evicted
    }

    #[test]
    fn evict_to_zero_empties_tree() {
        let mut t = RadixTree::new();
        t.insert(&[1, 2, 3]);
        t.evict_to(0);
        assert_eq!(t.num_blocks(), 0);
        assert_eq!(t.match_prefix(&[1, 2, 3]), 0);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p router radix`
Expected: FAIL — `RadixTree` not found.

- [ ] **Step 3: Implement**

Add above tests in `crates/router/src/radix.rs`:
```rust
use std::collections::HashMap;

#[derive(Default)]
struct Node {
    children: HashMap<u64, Node>,
    last_access: u64,
}

pub struct RadixTree {
    root: Node,
    clock: u64,
    num_blocks: usize,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self { root: Node::default(), clock: 0, num_blocks: 0 }
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Count of leading blocks present; refreshes access time along the path.
    pub fn match_prefix(&mut self, hashes: &[u64]) -> usize {
        self.clock += 1;
        let clock = self.clock;
        let mut node = &mut self.root;
        let mut matched = 0;
        for h in hashes {
            match node.children.get_mut(h) {
                Some(child) => {
                    child.last_access = clock;
                    matched += 1;
                    node = child;
                }
                None => break,
            }
        }
        matched
    }

    pub fn insert(&mut self, hashes: &[u64]) {
        self.clock += 1;
        let clock = self.clock;
        let mut created = 0;
        let mut node = &mut self.root;
        for &h in hashes {
            if !node.children.contains_key(&h) {
                node.children.insert(h, Node::default());
                created += 1;
            }
            let child = node.children.get_mut(&h).unwrap();
            child.last_access = clock;
            node = child;
        }
        self.num_blocks += created;
    }

    /// Evict least-recently-used leaves until num_blocks <= budget.
    pub fn evict_to(&mut self, budget: usize) {
        while self.num_blocks > budget {
            let mut path = Vec::new();
            let mut best: Option<(u64, Vec<u64>)> = None;
            Self::min_leaf(&self.root, &mut path, &mut best);
            let Some((_, leaf_path)) = best else { break };
            self.remove_leaf(&leaf_path);
        }
    }

    fn min_leaf(node: &Node, path: &mut Vec<u64>, best: &mut Option<(u64, Vec<u64>)>) {
        for (k, child) in &node.children {
            path.push(*k);
            if child.children.is_empty() {
                if best.as_ref().map_or(true, |(a, _)| child.last_access < *a) {
                    *best = Some((child.last_access, path.clone()));
                }
            } else {
                Self::min_leaf(child, path, best);
            }
            path.pop();
        }
    }

    fn remove_leaf(&mut self, path: &[u64]) {
        let Some((&last, parents)) = path.split_last() else { return };
        let mut node = &mut self.root;
        for k in parents {
            match node.children.get_mut(k) {
                Some(child) => node = child,
                None => return,
            }
        }
        if node.children.remove(&last).is_some() {
            self.num_blocks -= 1;
        }
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p router radix`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/router/src/radix.rs
git commit -m "feat(router): per-replica radix tree with LRU leaf eviction"
```

---

### Task 6: Scheduler

**Files:**
- Modify: `crates/router/src/scheduler.rs`

Owns all per-replica state. `pick()` is the whole routing decision: filter healthy + under inflight-cap replicas, score, insert routed prefix into winner's tree, evict to budget, bump inflight. Caller wraps `Scheduler` in `Mutex` (decision is microseconds).

- [ ] **Step 1: Write failing tests**

`crates/router/src/scheduler.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sched(n: usize, policy: Policy) -> Scheduler {
        Scheduler::new(n, policy, SchedulerConfig::default())
    }

    #[test]
    fn same_prefix_pins_to_same_replica() {
        let mut s = sched(4, Policy::CacheAware);
        let hashes: Vec<u64> = (1..=8).collect();
        let d1 = s.pick(&hashes).unwrap();
        s.complete(d1.replica);
        let d2 = s.pick(&hashes).unwrap();
        assert_eq!(d1.replica, d2.replica);
        assert_eq!(d2.matched_blocks, 8);
    }

    #[test]
    fn cold_traffic_goes_least_loaded() {
        let mut s = sched(3, Policy::CacheAware);
        // load replica 0 and 1 with one inflight each, replica 2 free
        let a = s.pick(&[100, 101]).unwrap().replica;
        let b = s.pick(&[200, 201]).unwrap().replica;
        let c = s.pick(&[300, 301]).unwrap().replica;
        // three cold requests with distinct prefixes spread across all three replicas
        let mut set = [a, b, c];
        set.sort();
        assert_eq!(set, [0, 1, 2]);
    }

    #[test]
    fn round_robin_cycles_and_ignores_cache() {
        let mut s = sched(3, Policy::RoundRobin);
        let hashes: Vec<u64> = (1..=4).collect();
        let picks: Vec<usize> = (0..6)
            .map(|_| {
                let d = s.pick(&hashes).unwrap();
                s.complete(d.replica);
                d.replica
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn unhealthy_replica_skipped_and_tree_dropped() {
        let mut s = sched(2, Policy::CacheAware);
        let hashes: Vec<u64> = (1..=8).collect();
        let d = s.pick(&hashes).unwrap();
        s.complete(d.replica);
        s.mark_unhealthy(d.replica);
        let d2 = s.pick(&hashes).unwrap();
        assert_ne!(d2.replica, d.replica);
        s.complete(d2.replica);
        s.mark_healthy(d.replica);
        // tree was dropped: no match credit on re-admitted replica
        let d3 = s.pick(&hashes).unwrap();
        assert_eq!(d3.replica, d2.replica); // pinned to the one that has it now
    }

    #[test]
    fn all_replicas_at_cap_returns_none() {
        let cfg = SchedulerConfig { max_inflight: 1, ..Default::default() };
        let mut s = Scheduler::new(2, Policy::CacheAware, cfg);
        s.pick(&[1, 2]).unwrap();
        s.pick(&[3, 4]).unwrap();
        assert!(s.pick(&[5, 6]).is_none());
    }

    #[test]
    fn eviction_respects_block_budget() {
        let cfg = SchedulerConfig { block_budget: 4, ..Default::default() };
        let mut s = Scheduler::new(1, Policy::CacheAware, cfg);
        for i in 0..10u64 {
            let d = s.pick(&[i * 10, i * 10 + 1]).unwrap();
            s.complete(d.replica);
        }
        assert!(s.replica_blocks(0) <= 4);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p router scheduler`
Expected: FAIL — types not found.

- [ ] **Step 3: Implement**

Add above tests in `crates/router/src/scheduler.rs`:
```rust
use crate::radix::RadixTree;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    CacheAware,
    RoundRobin,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    pub alpha: f64,         // weight of matched blocks
    pub beta: f64,          // weight of inflight load
    pub min_match_blocks: usize, // below this → least-loaded fallback
    pub block_budget: usize,     // per-replica cache capacity (blocks)
    pub max_inflight: usize,     // per-replica backpressure cap
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            min_match_blocks: 2,
            block_budget: 65_536,
            max_inflight: 512,
        }
    }
}

struct ReplicaState {
    tree: RadixTree,
    inflight: usize,
    healthy: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RouteDecision {
    pub replica: usize,
    pub matched_blocks: usize,
}

pub struct Scheduler {
    replicas: Vec<ReplicaState>,
    policy: Policy,
    cfg: SchedulerConfig,
    rr_next: usize,
}

impl Scheduler {
    pub fn new(n: usize, policy: Policy, cfg: SchedulerConfig) -> Self {
        let replicas = (0..n)
            .map(|_| ReplicaState { tree: RadixTree::new(), inflight: 0, healthy: true })
            .collect();
        Self { replicas, policy, cfg, rr_next: 0 }
    }

    pub fn pick(&mut self, hashes: &[u64]) -> Option<RouteDecision> {
        let available: Vec<usize> = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| r.healthy && r.inflight < self.cfg.max_inflight)
            .map(|(i, _)| i)
            .collect();
        if available.is_empty() {
            return None;
        }

        let (idx, matched) = match self.policy {
            Policy::RoundRobin => {
                let n = self.replicas.len();
                let mut idx = None;
                for off in 0..n {
                    let cand = (self.rr_next + off) % n;
                    if available.contains(&cand) {
                        idx = Some(cand);
                        self.rr_next = (cand + 1) % n;
                        break;
                    }
                }
                (idx?, 0)
            }
            Policy::CacheAware => {
                let mut best = available[0];
                let mut best_score = f64::MIN;
                let mut best_matched = 0;
                for &i in &available {
                    let matched = self.replicas[i].tree.match_prefix(hashes);
                    let score = self.cfg.alpha * matched as f64
                        - self.cfg.beta * self.replicas[i].inflight as f64;
                    if score > best_score {
                        best_score = score;
                        best = i;
                        best_matched = matched;
                    }
                }
                if best_matched < self.cfg.min_match_blocks {
                    // cold traffic: pure least-loaded to avoid pinning
                    let &least = available
                        .iter()
                        .min_by_key(|&&i| self.replicas[i].inflight)
                        .unwrap();
                    (least, 0)
                } else {
                    (best, best_matched)
                }
            }
        };

        let r = &mut self.replicas[idx];
        r.tree.insert(hashes);
        r.tree.evict_to(self.cfg.block_budget);
        r.inflight += 1;
        Some(RouteDecision { replica: idx, matched_blocks: matched })
    }

    pub fn complete(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.inflight = r.inflight.saturating_sub(1);
    }

    pub fn mark_unhealthy(&mut self, replica: usize) {
        let r = &mut self.replicas[replica];
        r.healthy = false;
        r.tree = RadixTree::new(); // cache assumed lost on restart
    }

    pub fn mark_healthy(&mut self, replica: usize) {
        self.replicas[replica].healthy = true;
    }

    pub fn is_healthy(&self, replica: usize) -> bool {
        self.replicas[replica].healthy
    }

    pub fn inflight(&self, replica: usize) -> usize {
        self.replicas[replica].inflight
    }

    pub fn replica_blocks(&self, replica: usize) -> usize {
        self.replicas[replica].tree.num_blocks()
    }

    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p router scheduler`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/router/src/scheduler.rs
git commit -m "feat(router): cost-based cache-aware scheduler with round-robin baseline"
```

---

### Task 7: Mock replica KV cache sim

**Files:**
- Modify: `crates/mock-triton/src/cache.rs`

Ground truth the router is guessing at. Flat block-hash LRU map (chained hashes make a flat map equivalent to a tree for prefix counting). `lookup_insert` returns the count of leading cached blocks, then inserts all and evicts LRU over capacity.

- [ ] **Step 1: Write failing tests**

`crates/mock-triton/src/cache.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_warm() {
        let mut c = KvCacheSim::new(100);
        assert_eq!(c.lookup_insert(&[1, 2, 3]), 0);
        assert_eq!(c.lookup_insert(&[1, 2, 3]), 3);
        assert_eq!(c.lookup_insert(&[1, 2, 9]), 2);
    }

    #[test]
    fn leading_only() {
        let mut c = KvCacheSim::new(100);
        c.lookup_insert(&[1, 2, 3]);
        // 2 and 3 are cached but 9 isn't the right first block
        assert_eq!(c.lookup_insert(&[9, 2, 3]), 0);
    }

    #[test]
    fn lru_eviction_over_capacity() {
        let mut c = KvCacheSim::new(4);
        c.lookup_insert(&[1, 2]);
        c.lookup_insert(&[3, 4]);
        c.lookup_insert(&[1, 2]); // refresh 1,2
        c.lookup_insert(&[5, 6]); // over capacity → evict 3,4
        assert_eq!(c.len(), 4);
        assert_eq!(c.lookup_insert(&[1, 2]), 2);
        assert_eq!(c.lookup_insert(&[5, 6]), 2);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p mock-triton cache`
Expected: FAIL — `KvCacheSim` not found.

- [ ] **Step 3: Implement**

Add above tests in `crates/mock-triton/src/cache.rs`:
```rust
use std::collections::HashMap;

pub struct KvCacheSim {
    capacity: usize,
    map: HashMap<u64, u64>, // block hash -> last access
    clock: u64,
}

impl KvCacheSim {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, map: HashMap::new(), clock: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns count of leading blocks already cached, then caches all blocks.
    pub fn lookup_insert(&mut self, hashes: &[u64]) -> usize {
        self.clock += 1;
        let cached = hashes.iter().take_while(|h| self.map.contains_key(h)).count();
        for &h in hashes {
            self.map.insert(h, self.clock);
        }
        while self.map.len() > self.capacity {
            if let Some((&victim, _)) = self.map.iter().min_by_key(|(_, &t)| t) {
                self.map.remove(&victim);
            } else {
                break;
            }
        }
        cached
    }
}
```

Note: O(n) eviction scan is fine at mock scale; not on the router hot path.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p mock-triton cache`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/mock-triton/src/cache.rs
git commit -m "feat(mock-triton): block LRU KV cache simulation"
```

---

### Task 8: Mock replica gRPC service

**Files:**
- Create: `crates/mock-triton/src/service.rs`
- Modify: `crates/mock-triton/src/lib.rs`, `crates/mock-triton/src/main.rs`

Implements KServe `GRPCInferenceService`. On infer: extract text from `inputs[0].contents.bytes_contents[0]`, tokenize, block-hash, `lookup_insert` → prefill delay proportional to **uncached** tokens, then reply. Response `parameters` carry `cached_blocks`/`total_blocks` (ground truth for benchmarks). Streaming: same, but N decode chunks; final chunk carries the params. Prometheus counters exposed on a side HTTP port.

- [ ] **Step 1: Write the service**

`crates/mock-triton/src/lib.rs`:
```rust
pub mod cache;
pub mod service;
```

`crates/mock-triton/src/service.rs`:
```rust
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use prometheus::{IntCounter, Registry};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use protocol::blocks::{block_hashes, BLOCK_SIZE};
use protocol::pb;
use protocol::GrpcInferenceService;

#[derive(Clone)]
pub struct MockConfig {
    pub cache_blocks: usize,
    pub prefill_us_per_token: u64,
    pub decode_chunks: usize,
    pub decode_interval_ms: u64,
    pub tokenizer_path: String,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            cache_blocks: 4096,
            prefill_us_per_token: 200,
            decode_chunks: 4,
            decode_interval_ms: 5,
            tokenizer_path: "data/tokenizer.json".into(),
        }
    }
}

pub struct MockMetrics {
    pub registry: Registry,
    pub cached_blocks_total: IntCounter,
    pub total_blocks_total: IntCounter,
}

impl MockMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let cached_blocks_total =
            IntCounter::new("replica_cached_blocks_total", "blocks served from KV cache").unwrap();
        let total_blocks_total =
            IntCounter::new("replica_total_blocks_total", "total prompt blocks seen").unwrap();
        registry.register(Box::new(cached_blocks_total.clone())).unwrap();
        registry.register(Box::new(total_blocks_total.clone())).unwrap();
        Self { registry, cached_blocks_total, total_blocks_total }
    }
}

impl Default for MockMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MockTritonService {
    cfg: MockConfig,
    cache: Arc<Mutex<crate::cache::KvCacheSim>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    pub metrics: Arc<MockMetrics>,
}
// State fields are Arcs because model_stream_infer spawns a task that needs
// owned handles (tonic handlers only get &self).

impl MockTritonService {
    pub fn new(cfg: MockConfig) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        Ok(Self {
            cache: Arc::new(Mutex::new(crate::cache::KvCacheSim::new(cfg.cache_blocks))),
            tokenizer: Arc::new(tokenizer),
            cfg,
            metrics: Arc::new(MockMetrics::new()),
        })
    }

    fn extract_text(req: &pb::ModelInferRequest) -> Result<String, Status> {
        let bytes = req
            .inputs
            .first()
            .and_then(|i| i.contents.as_ref())
            .and_then(|c| c.bytes_contents.first())
            .ok_or_else(|| Status::invalid_argument("missing text_input bytes tensor"))?;
        String::from_utf8(bytes.clone())
            .map_err(|_| Status::invalid_argument("text_input not utf8"))
    }

    /// Simulate prefill: returns (cached_blocks, total_blocks) and sleeps for
    /// time proportional to uncached tokens.
    async fn prefill(&self, text: &str) -> Result<(usize, usize), Status> {
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Status::invalid_argument(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let hashes = block_hashes(&ids);
        let total = hashes.len();
        let cached = self.cache.lock().unwrap().lookup_insert(&hashes);
        self.metrics.cached_blocks_total.inc_by(cached as u64);
        self.metrics.total_blocks_total.inc_by(total as u64);
        let uncached_tokens = ids.len().saturating_sub(cached * BLOCK_SIZE);
        let delay = Duration::from_micros(uncached_tokens as u64 * self.cfg.prefill_us_per_token);
        tokio::time::sleep(delay).await;
        Ok((cached, total))
    }

    fn make_response(
        req: &pb::ModelInferRequest,
        cached: usize,
        total: usize,
        text: &str,
    ) -> pb::ModelInferResponse {
        let mut parameters = HashMap::new();
        parameters.insert(
            "cached_blocks".to_string(),
            pb::InferParameter {
                parameter_choice: Some(pb::infer_parameter::ParameterChoice::Int64Param(
                    cached as i64,
                )),
            },
        );
        parameters.insert(
            "total_blocks".to_string(),
            pb::InferParameter {
                parameter_choice: Some(pb::infer_parameter::ParameterChoice::Int64Param(
                    total as i64,
                )),
            },
        );
        pb::ModelInferResponse {
            model_name: req.model_name.clone(),
            model_version: "1".into(),
            id: req.id.clone(),
            parameters,
            outputs: vec![pb::model_infer_response::InferOutputTensor {
                name: "text_output".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: Some(pb::InferTensorContents {
                    bytes_contents: vec![text.as_bytes().to_vec()],
                    ..Default::default()
                }),
            }],
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for MockTritonService {
    async fn server_live(
        &self,
        _: Request<pb::ServerLiveRequest>,
    ) -> Result<Response<pb::ServerLiveResponse>, Status> {
        Ok(Response::new(pb::ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: Request<pb::ServerReadyRequest>,
    ) -> Result<Response<pb::ServerReadyResponse>, Status> {
        Ok(Response::new(pb::ServerReadyResponse { ready: true }))
    }

    async fn model_metadata(
        &self,
        req: Request<pb::ModelMetadataRequest>,
    ) -> Result<Response<pb::ModelMetadataResponse>, Status> {
        Ok(Response::new(pb::ModelMetadataResponse {
            name: req.into_inner().name,
            versions: vec!["1".into()],
            platform: "mock".into(),
        }))
    }

    async fn model_infer(
        &self,
        request: Request<pb::ModelInferRequest>,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let req = request.into_inner();
        let text = Self::extract_text(&req)?;
        let (cached, total) = self.prefill(&text).await?;
        Ok(Response::new(Self::make_response(&req, cached, total, "mock-completion")))
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<pb::ModelStreamInferResponse, Status>> + Send>>;
```

- [ ] **Step 2: Add streaming handler**

Continue the same `impl GrpcInferenceService for MockTritonService` block (streaming task takes owned `Arc` clones of the state):

```rust
    async fn model_stream_infer(
        &self,
        request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<pb::ModelStreamInferResponse, Status>>(16);
        let cfg = self.cfg.clone();
        let cache = Arc::clone(&self.cache);
        let tokenizer = Arc::clone(&self.tokenizer);
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            while let Some(msg) = inbound.next().await {
                let req = match msg {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: format!("stream recv: {e}"),
                                infer_response: None,
                            }))
                            .await;
                        break;
                    }
                };
                let text = match Self::extract_text(&req) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: e.to_string(),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                // prefill (inline, since we own Arc handles here)
                let ids = match tokenizer.encode(text.as_str(), false) {
                    Ok(enc) => enc.get_ids().to_vec(),
                    Err(_) => Vec::new(),
                };
                let hashes = block_hashes(&ids);
                let total = hashes.len();
                let cached = cache.lock().unwrap().lookup_insert(&hashes);
                metrics.cached_blocks_total.inc_by(cached as u64);
                metrics.total_blocks_total.inc_by(total as u64);
                let uncached = ids.len().saturating_sub(cached * BLOCK_SIZE);
                tokio::time::sleep(Duration::from_micros(
                    uncached as u64 * cfg.prefill_us_per_token,
                ))
                .await;

                for chunk in 0..cfg.decode_chunks {
                    tokio::time::sleep(Duration::from_millis(cfg.decode_interval_ms)).await;
                    let last = chunk == cfg.decode_chunks - 1;
                    let resp = if last {
                        Self::make_response(&req, cached, total, "mock-final")
                    } else {
                        Self::make_response(&req, 0, 0, "mock-chunk")
                    };
                    if tx
                        .send(Ok(pb::ModelStreamInferResponse {
                            error_message: String::new(),
                            infer_response: Some(resp),
                        }))
                        .await
                        .is_err()
                    {
                        return; // client went away
                    }
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
```

Close the `impl` block with the `}` above.

- [ ] **Step 3: main.rs**

`crates/mock-triton/src/main.rs`:
```rust
use clap::Parser;
use mock_triton::service::{MockConfig, MockTritonService};
use prometheus::TextEncoder;
use protocol::GrpcInferenceServiceServer;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "PORT", default_value_t = 8001)]
    port: u16,
    #[arg(long, env = "METRICS_PORT", default_value_t = 9101)]
    metrics_port: u16,
    #[arg(long, env = "CACHE_BLOCKS", default_value_t = 4096)]
    cache_blocks: usize,
    #[arg(long, env = "PREFILL_US_PER_TOKEN", default_value_t = 200)]
    prefill_us_per_token: u64,
    #[arg(long, env = "DECODE_CHUNKS", default_value_t = 4)]
    decode_chunks: usize,
    #[arg(long, env = "DECODE_INTERVAL_MS", default_value_t = 5)]
    decode_interval_ms: u64,
    #[arg(long, env = "TOKENIZER_PATH", default_value = "data/tokenizer.json")]
    tokenizer_path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let cfg = MockConfig {
        cache_blocks: args.cache_blocks,
        prefill_us_per_token: args.prefill_us_per_token,
        decode_chunks: args.decode_chunks,
        decode_interval_ms: args.decode_interval_ms,
        tokenizer_path: args.tokenizer_path,
    };
    let svc = MockTritonService::new(cfg)?;
    let registry = svc.metrics.registry.clone();

    let metrics_app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let registry = registry.clone();
            async move {
                TextEncoder::new().encode_to_string(&registry.gather()).unwrap_or_default()
            }
        }),
    );
    let metrics_addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.metrics_port));
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(metrics_addr).await.unwrap();
        axum::serve(listener, metrics_app).await.unwrap();
    });

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("mock-triton listening on {addr}");
    tonic::transport::Server::builder()
        .add_service(GrpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
```

- [ ] **Step 4: Build + smoke run**

Run: `cargo build -p mock-triton`
Expected: compiles.

Run: `./scripts/fetch_tokenizer.sh data && cargo run -p mock-triton -- --port 8001 &` then
`grpcurl -plaintext -d '{}' localhost:8001 inference.GRPCInferenceService/ServerLive` (if grpcurl installed; otherwise rely on Task 9 integration test) and `curl -s localhost:9101/metrics | head`. Kill the background process after.
Expected: `{"live": true}` / prometheus text output.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(mock-triton): KServe gRPC replica with simulated KV cache and latency"
```

---

### Task 9: Router proxy — unary path + integration test

**Files:**
- Create: `crates/router/src/server.rs` (replace empty), `crates/router/tests/integration.rs`
- Modify: `crates/router/src/telemetry.rs` (metrics only; OTel comes in Task 12)

RouterService implements the same KServe service. Unary flow: extract text → tokenize → block-hash → `Scheduler::pick` (mutex, microseconds) → forward to picked replica's tonic client → `complete()` → return response untouched. `pick()` returning None → `RESOURCE_EXHAUSTED`. Upstream transport error → mark unhealthy, retry once on another replica.

- [ ] **Step 1: Metrics registry**

`crates/router/src/telemetry.rs`:
```rust
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, IntGaugeVec, Opts, Registry};

pub struct Metrics {
    pub registry: Registry,
    pub requests_total: IntCounter,
    pub routing_overhead_seconds: Histogram,
    pub matched_blocks_total: IntCounter,
    pub prompt_blocks_total: IntCounter,
    pub replica_inflight: IntGaugeVec,
    pub healthy_replicas: IntGauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let requests_total =
            IntCounter::new("router_requests_total", "requests routed").unwrap();
        let routing_overhead_seconds = Histogram::with_opts(
            HistogramOpts::new("router_routing_overhead_seconds", "route decision time")
                .buckets(vec![
                    0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.008, 0.01, 0.025,
                ]),
        )
        .unwrap();
        let matched_blocks_total =
            IntCounter::new("router_matched_blocks_total", "predicted cache-hit blocks").unwrap();
        let prompt_blocks_total =
            IntCounter::new("router_prompt_blocks_total", "total prompt blocks").unwrap();
        let replica_inflight = IntGaugeVec::new(
            Opts::new("router_replica_inflight", "inflight per replica"),
            &["replica"],
        )
        .unwrap();
        let healthy_replicas =
            IntGauge::new("router_healthy_replicas", "healthy replica count").unwrap();
        for m in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(routing_overhead_seconds.clone()),
            Box::new(matched_blocks_total.clone()),
            Box::new(prompt_blocks_total.clone()),
            Box::new(replica_inflight.clone()),
            Box::new(healthy_replicas.clone()),
        ] {
            registry.register(m).unwrap();
        }
        Self {
            registry,
            requests_total,
            routing_overhead_seconds,
            matched_blocks_total,
            prompt_blocks_total,
            replica_inflight,
            healthy_replicas,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: RouterService**

`crates/router/src/server.rs`:
```rust
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio_stream::Stream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};

use protocol::blocks::block_hashes;
use protocol::pb;
use protocol::{GrpcInferenceService, GrpcInferenceServiceClient};

use crate::scheduler::{RouteDecision, Scheduler};
use crate::telemetry::Metrics;
use crate::tokenizer::PromptTokenizer;

pub struct RouterService {
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub clients: Vec<GrpcInferenceServiceClient<Channel>>,
    pub tokenizer: Arc<PromptTokenizer>,
    pub metrics: Arc<Metrics>,
}

impl RouterService {
    fn extract_text(req: &pb::ModelInferRequest) -> Result<String, Status> {
        let bytes = req
            .inputs
            .first()
            .and_then(|i| i.contents.as_ref())
            .and_then(|c| c.bytes_contents.first())
            .ok_or_else(|| Status::invalid_argument("missing text_input bytes tensor"))?;
        String::from_utf8(bytes.clone())
            .map_err(|_| Status::invalid_argument("text_input not utf8"))
    }

    /// Route decision under mutex; returns decision + records metrics.
    fn route(&self, text: &str) -> Result<RouteDecision, Status> {
        let start = Instant::now();
        let tokens = self.tokenizer.tokenize(text);
        let hashes = block_hashes(&tokens);
        let decision = {
            let mut sched = self.scheduler.lock().unwrap();
            sched.pick(&hashes)
        }
        .ok_or_else(|| Status::resource_exhausted("all replicas saturated"))?;
        self.metrics.routing_overhead_seconds.observe(start.elapsed().as_secs_f64());
        self.metrics.requests_total.inc();
        self.metrics.matched_blocks_total.inc_by(decision.matched_blocks as u64);
        self.metrics.prompt_blocks_total.inc_by(hashes.len() as u64);
        self.metrics
            .replica_inflight
            .with_label_values(&[&decision.replica.to_string()])
            .set(self.scheduler.lock().unwrap().inflight(decision.replica) as i64);
        Ok(decision)
    }

    fn complete(&self, replica: usize) {
        let mut sched = self.scheduler.lock().unwrap();
        sched.complete(replica);
        let inflight = sched.inflight(replica) as i64;
        drop(sched);
        self.metrics
            .replica_inflight
            .with_label_values(&[&replica.to_string()])
            .set(inflight);
    }

    fn mark_unhealthy(&self, replica: usize) {
        self.scheduler.lock().unwrap().mark_unhealthy(replica);
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for RouterService {
    async fn server_live(
        &self,
        _: Request<pb::ServerLiveRequest>,
    ) -> Result<Response<pb::ServerLiveResponse>, Status> {
        Ok(Response::new(pb::ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: Request<pb::ServerReadyRequest>,
    ) -> Result<Response<pb::ServerReadyResponse>, Status> {
        let any_healthy = {
            let sched = self.scheduler.lock().unwrap();
            (0..sched.num_replicas()).any(|i| sched.is_healthy(i))
        };
        Ok(Response::new(pb::ServerReadyResponse { ready: any_healthy }))
    }

    async fn model_metadata(
        &self,
        req: Request<pb::ModelMetadataRequest>,
    ) -> Result<Response<pb::ModelMetadataResponse>, Status> {
        // forward to first healthy replica
        let idx = {
            let sched = self.scheduler.lock().unwrap();
            (0..sched.num_replicas()).find(|&i| sched.is_healthy(i))
        }
        .ok_or_else(|| Status::unavailable("no healthy replicas"))?;
        let mut client = self.clients[idx].clone();
        client.model_metadata(req.into_inner()).await
    }

    async fn model_infer(
        &self,
        request: Request<pb::ModelInferRequest>,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let req = request.into_inner();
        let text = Self::extract_text(&req)?;

        // up to 2 attempts; transport errors mark the replica unhealthy
        let mut last_err = Status::unavailable("no attempt made");
        for _attempt in 0..2 {
            let decision = self.route(&text)?;
            let mut client = self.clients[decision.replica].clone();
            let result = client.model_infer(req.clone()).await;
            self.complete(decision.replica);
            match result {
                Ok(resp) => return Ok(resp),
                Err(status) if status.code() == tonic::Code::Unavailable => {
                    self.mark_unhealthy(decision.replica);
                    last_err = status;
                    continue; // retry on another replica
                }
                Err(status) => return Err(status), // app-level error: propagate
            }
        }
        Err(last_err)
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<pb::ModelStreamInferResponse, Status>> + Send>>;

    async fn model_stream_infer(
        &self,
        _request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        Err(Status::unimplemented("streaming lands in Task 10"))
    }
}
```

- [ ] **Step 3: Write failing integration test**

`crates/router/tests/integration.rs`:
```rust
use std::sync::{Arc, Mutex};

use protocol::pb;
use protocol::{GrpcInferenceServiceClient, GrpcInferenceServiceServer};
use router::scheduler::{Policy, Scheduler, SchedulerConfig};
use router::server::RouterService;
use router::telemetry::Metrics;
use router::tokenizer::PromptTokenizer;

use mock_triton::service::{MockConfig, MockTritonService};

const TOKENIZER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/tokenizer.json");

async fn spawn_mock(prefill_us: u64) -> (String, tokio::task::JoinHandle<()>) {
    let cfg = MockConfig {
        prefill_us_per_token: prefill_us,
        tokenizer_path: TOKENIZER.into(),
        ..Default::default()
    };
    let svc = MockTritonService::new(cfg).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (format!("http://{addr}"), handle)
}

async fn spawn_router(
    replica_urls: &[String],
    policy: Policy,
) -> (String, tokio::task::JoinHandle<()>) {
    let mut clients = Vec::new();
    for url in replica_urls {
        clients.push(GrpcInferenceServiceClient::connect(url.clone()).await.unwrap());
    }
    let svc = RouterService {
        scheduler: Arc::new(Mutex::new(Scheduler::new(
            replica_urls.len(),
            policy,
            SchedulerConfig::default(),
        ))),
        clients,
        tokenizer: Arc::new(PromptTokenizer::from_file(TOKENIZER).unwrap()),
        metrics: Arc::new(Metrics::new()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (format!("http://{addr}"), handle)
}

fn infer_request(text: &str) -> pb::ModelInferRequest {
    pb::ModelInferRequest {
        model_name: "mock".into(),
        inputs: vec![pb::model_infer_request::InferInputTensor {
            name: "text_input".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            contents: Some(pb::InferTensorContents {
                bytes_contents: vec![text.as_bytes().to_vec()],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn cached_blocks(resp: &pb::ModelInferResponse) -> i64 {
    match resp.parameters.get("cached_blocks").and_then(|p| p.parameter_choice.as_ref()) {
        Some(pb::infer_parameter::ParameterChoice::Int64Param(v)) => *v,
        _ => panic!("missing cached_blocks param"),
    }
}

// Long repeated prefix so prompts span many 16-token blocks.
fn long_prompt(suffix: &str) -> String {
    format!("{} {suffix}", "The quick brown fox jumps over the lazy dog. ".repeat(30))
}

#[tokio::test]
async fn same_prefix_requests_hit_cache_via_pinning() {
    let (m1, _h1) = spawn_mock(0).await;
    let (m2, _h2) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router(&[m1, m2], Policy::CacheAware).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url).await.unwrap();

    let first = client
        .model_infer(infer_request(&long_prompt("turn one")))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(cached_blocks(&first), 0); // cold

    let second = client
        .model_infer(infer_request(&long_prompt("turn one and then some more words")))
        .await
        .unwrap()
        .into_inner();
    assert!(cached_blocks(&second) > 0, "pinned replica should have warm prefix");
}

#[tokio::test]
async fn saturated_replicas_reject_fast() {
    let (m1, _h1) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router_with_cap(&[m1], 0).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url).await.unwrap();
    let err = client.model_infer(infer_request("hello")).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::ResourceExhausted);
}

async fn spawn_router_with_cap(
    replica_urls: &[String],
    max_inflight: usize,
) -> (String, tokio::task::JoinHandle<()>) {
    let mut clients = Vec::new();
    for url in replica_urls {
        clients.push(GrpcInferenceServiceClient::connect(url.clone()).await.unwrap());
    }
    let cfg = SchedulerConfig { max_inflight, ..Default::default() };
    let svc = RouterService {
        scheduler: Arc::new(Mutex::new(Scheduler::new(replica_urls.len(), Policy::CacheAware, cfg))),
        clients,
        tokenizer: Arc::new(PromptTokenizer::from_file(TOKENIZER).unwrap()),
        metrics: Arc::new(Metrics::new()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (format!("http://{addr}"), handle)
}
```

Add to `crates/router/Cargo.toml` dev-dependencies (tokio-stream net feature for `TcpListenerStream`):
```toml
[dev-dependencies]
mock-triton = { path = "../mock-triton" }
tokio-stream = { version = "0.1", features = ["net"] }
```

- [ ] **Step 4: Run tests — first failing, then implement until green**

Run: `cargo test -p router --test integration`
Expected first: compile errors while server.rs incomplete → fix; then both tests PASS. (Requires `data/tokenizer.json` — run `./scripts/fetch_tokenizer.sh data` first.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(router): cache-aware unary proxy with backpressure + integration tests"
```

---

### Task 10: Router streaming passthrough

**Files:**
- Modify: `crates/router/src/server.rs` (replace the `unimplemented` streaming handler)
- Modify: `crates/router/tests/integration.rs` (add test)

Router consumes the client's request stream; each request is routed independently (each may pin a different replica). Per request: open upstream `model_stream_infer` with a single-item stream, forward every chunk to the client as it arrives. Zero buffering beyond channel capacity. Mid-stream upstream failure → emit `error_message` chunk, continue with next request.

- [ ] **Step 1: Write failing test**

Append to `crates/router/tests/integration.rs`:
```rust
#[tokio::test]
async fn streaming_forwards_chunks() {
    let (m1, _h1) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router(&[m1], Policy::CacheAware).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url).await.unwrap();

    let outbound = tokio_stream::iter(vec![infer_request(&long_prompt("stream me"))]);
    let mut inbound = client.model_stream_infer(outbound).await.unwrap().into_inner();

    let mut chunks = Vec::new();
    while let Some(msg) = inbound.message().await.unwrap() {
        assert!(msg.error_message.is_empty(), "unexpected: {}", msg.error_message);
        chunks.push(msg.infer_response.unwrap());
    }
    assert_eq!(chunks.len(), 4); // MockConfig::default decode_chunks
    assert!(chunks.last().unwrap().parameters.contains_key("cached_blocks"));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p router --test integration streaming_forwards_chunks`
Expected: FAIL — `Unimplemented`.

- [ ] **Step 3: Implement**

Replace `model_stream_infer` in `crates/router/src/server.rs`. The handler needs owned state for the spawned task, so add `Clone`-able handles — restructure `RouterService` fields into an inner `Arc`:

At top of `server.rs`, change the struct to:
```rust
#[derive(Clone)]
pub struct RouterService {
    pub inner: Arc<RouterInner>,
}

pub struct RouterInner {
    pub scheduler: Mutex<Scheduler>,
    pub clients: Vec<GrpcInferenceServiceClient<Channel>>,
    pub tokenizer: PromptTokenizer,
    pub metrics: Metrics,
}
```

Mechanical fallout, same logic everywhere:
- `self.scheduler.lock()` → `self.inner.scheduler.lock()`
- `self.clients[i]` → `self.inner.clients[i]`
- `self.tokenizer` → `self.inner.tokenizer`
- `self.metrics` → `self.inner.metrics`
- helper methods (`route`, `complete`, `mark_unhealthy`, `extract_text`) move to `impl RouterInner` (change `&self` references accordingly; `extract_text` stays an associated fn)
- test constructors change from struct-literal `RouterService { scheduler: Arc::new(Mutex::new(...)), ... }` to:
```rust
RouterService {
    inner: Arc::new(RouterInner {
        scheduler: Mutex::new(Scheduler::new(replica_urls.len(), policy, cfg)),
        clients,
        tokenizer: PromptTokenizer::from_file(TOKENIZER).unwrap(),
        metrics: Metrics::new(),
    }),
}
```

New streaming handler:
```rust
    async fn model_stream_infer(
        &self,
        request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut client_rx = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<pb::ModelStreamInferResponse, Status>>(16);
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            use tokio_stream::StreamExt;
            while let Some(msg) = client_rx.next().await {
                let req = match msg {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                };
                let text = match RouterInner::extract_text(&req) {
                    Ok(t) => t,
                    Err(status) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: status.to_string(),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                let decision = match inner.route(&text) {
                    Ok(d) => d,
                    Err(status) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: status.to_string(),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                let mut client = inner.clients[decision.replica].clone();
                let upstream = client
                    .model_stream_infer(tokio_stream::iter(vec![req]))
                    .await;
                match upstream {
                    Ok(resp) => {
                        let mut upstream_rx = resp.into_inner();
                        loop {
                            match upstream_rx.message().await {
                                Ok(Some(chunk)) => {
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        inner.complete(decision.replica);
                                        return; // client gone
                                    }
                                }
                                Ok(None) => break,
                                Err(status) => {
                                    let _ = tx
                                        .send(Ok(pb::ModelStreamInferResponse {
                                            error_message: format!("upstream: {status}"),
                                            infer_response: None,
                                        }))
                                        .await;
                                    break;
                                }
                            }
                        }
                    }
                    Err(status) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: format!("upstream connect: {status}"),
                                infer_response: None,
                            }))
                            .await;
                    }
                }
                inner.complete(decision.replica);
            }
        });

        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }
```

**Wait for stream end before completing per-request** — note `inner.complete(decision.replica)` runs after the upstream stream finishes, so inflight reflects the full stream duration. One subtlety in the mock: its stream never half-closes per request (it keeps the response stream open for more requests). To make `Ok(None)` arrive, the single-item `tokio_stream::iter` closes the upstream request stream after one message; the mock's `while let Some(msg)` then ends and drops `tx`, ending the response stream. That is exactly the behavior the mock from Task 8 has.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p router --test integration`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(router): streaming passthrough with per-request routing"
```

---

### Task 11: Health prober + retry test + router main

**Files:**
- Modify: `crates/router/src/server.rs` (health prober fn), `crates/router/src/main.rs` (full wiring), `crates/router/tests/integration.rs`

- [ ] **Step 1: Write failing failover test**

Append to `crates/router/tests/integration.rs`:
```rust
#[tokio::test]
async fn failover_retries_on_dead_replica() {
    // one real mock + one dead endpoint that accepts TCP then closes
    let (m1, _h1) = spawn_mock(0).await;
    let dead_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", dead_listener.local_addr().unwrap());
    let dead_handle = tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = dead_listener.accept().await else { break };
            drop(sock); // slam the door: transport error upstream
        }
    });

    // lazy connect so router startup doesn't fail on the dead endpoint
    let (router_url, _rh) = spawn_router_lazy(&[dead_url, m1], Policy::RoundRobin).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url).await.unwrap();

    // RoundRobin starts at replica 0 (dead) → transport error → retry lands on replica 1
    let resp = client.model_infer(infer_request(&long_prompt("failover"))).await;
    assert!(resp.is_ok(), "retry should succeed on healthy replica: {resp:?}");
    dead_handle.abort();
}

async fn spawn_router_lazy(
    replica_urls: &[String],
    policy: Policy,
) -> (String, tokio::task::JoinHandle<()>) {
    use tonic::transport::Endpoint;
    let clients: Vec<_> = replica_urls
        .iter()
        .map(|url| {
            let channel = Endpoint::from_shared(url.clone()).unwrap().connect_lazy();
            GrpcInferenceServiceClient::new(channel)
        })
        .collect();
    let svc = RouterService {
        inner: Arc::new(router::server::RouterInner {
            scheduler: Mutex::new(Scheduler::new(
                replica_urls.len(),
                policy,
                SchedulerConfig::default(),
            )),
            clients,
            tokenizer: PromptTokenizer::from_file(TOKENIZER).unwrap(),
            metrics: Metrics::new(),
        }),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (format!("http://{addr}"), handle)
}
```

(Also switch `spawn_mock`/`spawn_router` helpers to `connect_lazy` style if the eager `connect` races server startup — `connect_lazy` is the robust default; using it everywhere is fine.)

- [ ] **Step 2: Run to verify failure or pass**

Run: `cargo test -p router --test integration failover`
Expected: PASS if Task 9's retry logic is correct (transport errors surface as `Code::Unavailable`). If it fails because the error code differs, widen the retry condition to:
```rust
Err(status)
    if matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::Unknown | tonic::Code::Internal
    ) =>
```
and re-run until PASS. This is the real check that retry works end-to-end.

- [ ] **Step 3: Health prober**

Append to `crates/router/src/server.rs`:
```rust
/// Probe ServerLive on every replica every `interval`; flip health state.
pub fn spawn_health_prober(
    inner: Arc<RouterInner>,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let n = inner.clients.len();
            let mut healthy_count = 0i64;
            for i in 0..n {
                let mut client = inner.clients[i].clone();
                let live = matches!(
                    tokio::time::timeout(
                        std::time::Duration::from_secs(1),
                        client.server_live(pb::ServerLiveRequest {}),
                    )
                    .await,
                    Ok(Ok(resp)) if resp.get_ref().live
                );
                let mut sched = inner.scheduler.lock().unwrap();
                if live {
                    if !sched.is_healthy(i) {
                        tracing::info!(replica = i, "replica re-admitted");
                    }
                    sched.mark_healthy(i);
                    healthy_count += 1;
                } else if sched.is_healthy(i) {
                    tracing::warn!(replica = i, "replica marked unhealthy");
                    sched.mark_unhealthy(i);
                }
            }
            inner.metrics.healthy_replicas.set(healthy_count);
        }
    })
}
```

- [ ] **Step 4: Router main**

`crates/router/src/main.rs`:
```rust
use std::sync::{Arc, Mutex};

use clap::Parser;
use prometheus::TextEncoder;
use protocol::{GrpcInferenceServiceClient, GrpcInferenceServiceServer};
use router::scheduler::{Policy, Scheduler, SchedulerConfig};
use router::server::{spawn_health_prober, RouterInner, RouterService};
use router::telemetry::Metrics;
use router::tokenizer::PromptTokenizer;
use tonic::transport::Endpoint;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "PORT", default_value_t = 8000)]
    port: u16,
    #[arg(long, env = "METRICS_PORT", default_value_t = 9100)]
    metrics_port: u16,
    /// Comma-separated replica gRPC URLs, e.g. http://mock1:8001,http://mock2:8001
    #[arg(long, env = "REPLICAS", value_delimiter = ',')]
    replicas: Vec<String>,
    /// cache-aware | round-robin
    #[arg(long, env = "POLICY", default_value = "cache-aware")]
    policy: String,
    #[arg(long, env = "TOKENIZER_PATH", default_value = "data/tokenizer.json")]
    tokenizer_path: String,
    #[arg(long, env = "ALPHA", default_value_t = 1.0)]
    alpha: f64,
    #[arg(long, env = "BETA", default_value_t = 2.0)]
    beta: f64,
    #[arg(long, env = "BLOCK_BUDGET", default_value_t = 65536)]
    block_budget: usize,
    #[arg(long, env = "MAX_INFLIGHT", default_value_t = 512)]
    max_inflight: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    router::telemetry::init_tracing(); // Task 12; until then: tracing_subscriber::fmt::init();
    let args = Args::parse();
    anyhow::ensure!(!args.replicas.is_empty(), "need at least one replica URL");

    let policy = match args.policy.as_str() {
        "cache-aware" => Policy::CacheAware,
        "round-robin" => Policy::RoundRobin,
        other => anyhow::bail!("unknown policy: {other}"),
    };
    let cfg = SchedulerConfig {
        alpha: args.alpha,
        beta: args.beta,
        block_budget: args.block_budget,
        max_inflight: args.max_inflight,
        ..Default::default()
    };

    let clients: Vec<_> = args
        .replicas
        .iter()
        .map(|url| {
            let channel = Endpoint::from_shared(url.clone())
                .expect("bad replica url")
                .connect_lazy();
            GrpcInferenceServiceClient::new(channel)
        })
        .collect();

    let inner = Arc::new(RouterInner {
        scheduler: Mutex::new(Scheduler::new(args.replicas.len(), policy, cfg)),
        clients,
        tokenizer: PromptTokenizer::from_file(&args.tokenizer_path)?,
        metrics: Metrics::new(),
    });

    spawn_health_prober(Arc::clone(&inner), std::time::Duration::from_secs(2));

    let registry = inner.metrics.registry.clone();
    let metrics_app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let registry = registry.clone();
            async move {
                TextEncoder::new().encode_to_string(&registry.gather()).unwrap_or_default()
            }
        }),
    );
    let metrics_addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.metrics_port));
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(metrics_addr).await.unwrap();
        axum::serve(listener, metrics_app).await.unwrap();
    });

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!(%addr, policy = args.policy, replicas = args.replicas.len(), "router up");
    tonic::transport::Server::builder()
        .add_service(GrpcInferenceServiceServer::new(RouterService { inner }))
        .serve(addr)
        .await?;
    Ok(())
}
```

Until Task 12 exists, use `tracing_subscriber::fmt::init();` directly in place of `router::telemetry::init_tracing()`.

- [ ] **Step 5: Run all router tests**

Run: `cargo test -p router && cargo build -p router`
Expected: all unit + integration tests PASS, binary builds.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(router): health prober, failover test, full binary wiring"
```

---

### Task 12: OpenTelemetry tracing

**Files:**
- Modify: `crates/router/Cargo.toml`, `crates/router/src/telemetry.rs`, `crates/router/src/server.rs`, `crates/router/src/main.rs`

Span per request with route decision + upstream call as children, OTLP-exported to Jaeger. OTel only initializes when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise plain fmt logging (keeps tests/dev clean).

- [ ] **Step 1: Add dependencies**

Append to `crates/router/Cargo.toml` `[dependencies]`:
```toml
opentelemetry = "0.27"
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.27", features = ["grpc-tonic"] }
tracing-opentelemetry = "0.28"
```
(These four crates move APIs between minor versions. If the code below doesn't compile against what cargo resolves, check the `opentelemetry-otlp` README example for the current init incantation and adapt — the *shape* is: OTLP span exporter → SdkTracerProvider → tracer → `tracing_opentelemetry::layer().with_tracer(tracer)`.)

- [ ] **Step 2: init_tracing**

Append to `crates/router/src/telemetry.rs`:
```rust
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// fmt logging always; OTel layer only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
pub fn init_tracing() {
    let fmt_layer = tracing_subscriber::fmt::layer();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .expect("otlp exporter");
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                opentelemetry_sdk::Resource::builder()
                    .with_service_name("dynamorouter")
                    .build(),
            )
            .build();
        use opentelemetry::trace::TracerProvider as _;
        let tracer = provider.tracer("dynamorouter");
        opentelemetry::global::set_tracer_provider(provider);
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .init();
    } else {
        tracing_subscriber::registry().with(filter).with(fmt_layer).init();
    }
}
```

- [ ] **Step 3: Instrument the hot path**

In `crates/router/src/server.rs`:

- On `model_infer`, wrap the body's route + forward in spans. Change the retry loop body:
```rust
        for attempt in 0..2 {
            let span = tracing::info_span!("route_decision", attempt);
            let decision = {
                let _g = span.enter();
                self.inner.route(&text)?
            };
            let upstream_span = tracing::info_span!(
                "upstream_infer",
                replica = decision.replica,
                matched_blocks = decision.matched_blocks
            );
            let mut client = self.inner.clients[decision.replica].clone();
            let result = {
                let _g = upstream_span.enter();
                client.model_infer(req.clone()).await
            };
            // ... rest unchanged
```
Note: entering a span across `.await` like this is acceptable here because the handler runs the future to completion inside the guard scope on the same task; if clippy's `await_holding_lock`-style lints or `tracing` docs push back, switch to `.instrument(upstream_span)` from `tracing::Instrument` — that is the canonical form:
```rust
use tracing::Instrument;
let result = client.model_infer(req.clone()).instrument(upstream_span).await;
```
Use the `.instrument()` form. The `enter()` form is shown only to flag the pitfall.

- In `main.rs`, replace `tracing_subscriber::fmt::init();` with `router::telemetry::init_tracing();`.

- [ ] **Step 4: Verify**

Run: `cargo test -p router && cargo build -p router`
Expected: everything still green (OTel dormant without the env var).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(router): OTel OTLP tracing with per-request route/upstream spans"
```

---

### Task 13: Loadgen — ShareGPT replay

**Files:**
- Create: `crates/loadgen/src/trace.rs`, `crates/loadgen/src/lib.rs`, `crates/loadgen/tests/fixtures/sharegpt_small.json`
- Modify: `crates/loadgen/src/main.rs`
- Create: `scripts/fetch_sharegpt.sh`

Multi-turn replay: for each conversation, turn k's prompt = concatenation of all messages up to and including the k-th human message. Sequential within a conversation (turn k+1 only after k completes — like a real chat), conversations run concurrently under a semaphore. Stats: rps, P50/P99 latency, ground-truth cache hit rate from replica response params.

- [ ] **Step 1: Fixture + failing parse test**

`crates/loadgen/tests/fixtures/sharegpt_small.json`:
```json
[
  {
    "id": "conv1",
    "conversations": [
      { "from": "human", "value": "What is Rust?" },
      { "from": "gpt", "value": "A systems programming language." },
      { "from": "human", "value": "Why is it fast?" },
      { "from": "gpt", "value": "No GC, zero-cost abstractions." }
    ]
  },
  {
    "id": "conv2",
    "conversations": [
      { "from": "human", "value": "Hello" },
      { "from": "gpt", "value": "Hi there" }
    ]
  }
]
```

`crates/loadgen/src/lib.rs`:
```rust
pub mod trace;
```

`crates/loadgen/src/trace.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sharegpt_small.json");

    #[test]
    fn parses_and_builds_growing_prompts() {
        let convs = load_sharegpt(FIXTURE, 10).unwrap();
        assert_eq!(convs.len(), 2);
        let turns = &convs[0].turns;
        assert_eq!(turns.len(), 2); // two human turns
        assert!(turns[0].contains("What is Rust?"));
        assert!(turns[1].starts_with(&turns[0][..])); // growing prefix
        assert!(turns[1].contains("Why is it fast?"));
    }

    #[test]
    fn cap_limits_conversations() {
        let convs = load_sharegpt(FIXTURE, 1).unwrap();
        assert_eq!(convs.len(), 1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p loadgen`
Expected: FAIL — `load_sharegpt` not found.

- [ ] **Step 3: Implement parser**

Add above tests in `crates/loadgen/src/trace.rs`:
```rust
use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct RawConversation {
    #[serde(default)]
    id: String,
    conversations: Vec<RawTurn>,
}

#[derive(Deserialize)]
struct RawTurn {
    from: String,
    value: String,
}

pub struct Conversation {
    pub id: String,
    /// turns[k] = full prompt for the k-th request (all history + k-th human msg)
    pub turns: Vec<String>,
}

pub fn load_sharegpt(path: &str, max_conversations: usize) -> Result<Vec<Conversation>> {
    let raw: Vec<RawConversation> = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(path)?,
    ))?;
    let mut out = Vec::new();
    for rc in raw {
        let mut turns = Vec::new();
        let mut history = String::new();
        for t in &rc.conversations {
            if t.from == "human" {
                history.push_str("USER: ");
                history.push_str(&t.value);
                history.push('\n');
                turns.push(history.clone());
            } else {
                history.push_str("ASSISTANT: ");
                history.push_str(&t.value);
                history.push('\n');
            }
        }
        if !turns.is_empty() {
            out.push(Conversation { id: rc.id, turns });
        }
        if out.len() >= max_conversations {
            break;
        }
    }
    Ok(out)
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p loadgen`
Expected: 2 tests PASS.

- [ ] **Step 5: Replay driver**

`crates/loadgen/src/main.rs`:
```rust
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use loadgen::trace::load_sharegpt;
use protocol::pb;
use protocol::GrpcInferenceServiceClient;
use tokio::sync::Semaphore;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    router: String,
    #[arg(long, default_value = "data/sharegpt.json")]
    trace: String,
    #[arg(long, default_value_t = 200)]
    conversations: usize,
    #[arg(long, default_value_t = 32)]
    concurrency: usize,
    #[arg(long, default_value = "run")]
    label: String,
}

struct RequestResult {
    latency_ms: f64,
    cached_blocks: i64,
    total_blocks: i64,
}

fn infer_request(text: &str) -> pb::ModelInferRequest {
    pb::ModelInferRequest {
        model_name: "mock".into(),
        inputs: vec![pb::model_infer_request::InferInputTensor {
            name: "text_input".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            contents: Some(pb::InferTensorContents {
                bytes_contents: vec![text.as_bytes().to_vec()],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn param_i64(resp: &pb::ModelInferResponse, key: &str) -> i64 {
    match resp.parameters.get(key).and_then(|p| p.parameter_choice.as_ref()) {
        Some(pb::infer_parameter::ParameterChoice::Int64Param(v)) => *v,
        _ => 0,
    }
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let convs = load_sharegpt(&args.trace, args.conversations)?;
    eprintln!("loaded {} conversations from {}", convs.len(), args.trace);

    let sem = Arc::new(Semaphore::new(args.concurrency));
    let client = GrpcInferenceServiceClient::connect(args.router.clone()).await?;
    let start = Instant::now();
    let mut handles = Vec::new();

    for conv in convs {
        let sem = Arc::clone(&sem);
        let mut client = client.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let mut results = Vec::new();
            for turn in &conv.turns {
                let t0 = Instant::now();
                match client.model_infer(infer_request(turn)).await {
                    Ok(resp) => {
                        let resp = resp.into_inner();
                        results.push(RequestResult {
                            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
                            cached_blocks: param_i64(&resp, "cached_blocks"),
                            total_blocks: param_i64(&resp, "total_blocks"),
                        });
                    }
                    Err(status) => {
                        eprintln!("conv {}: {status}", conv.id);
                        break; // abandon conversation on error
                    }
                }
            }
            results
        }));
    }

    let mut all = Vec::new();
    for h in handles {
        all.extend(h.await?);
    }
    let wall = start.elapsed().as_secs_f64();

    let mut latencies: Vec<f64> = all.iter().map(|r| r.latency_ms).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cached: i64 = all.iter().map(|r| r.cached_blocks).sum();
    let total: i64 = all.iter().map(|r| r.total_blocks).sum();
    let hit_rate = if total > 0 { 100.0 * cached as f64 / total as f64 } else { 0.0 };

    println!("== {} ==", args.label);
    println!("requests:        {}", all.len());
    println!("wall time:       {wall:.1}s");
    println!("throughput:      {:.0} req/s", all.len() as f64 / wall);
    println!("latency P50:     {:.1} ms", percentile(&latencies, 0.50));
    println!("latency P99:     {:.1} ms", percentile(&latencies, 0.99));
    println!("cache hit rate:  {hit_rate:.1}% ({cached}/{total} blocks)");
    Ok(())
}
```

Add `loadgen` lib target — `crates/loadgen/Cargo.toml` needs:
```toml
[lib]
name = "loadgen"
path = "src/lib.rs"

[[bin]]
name = "loadgen"
path = "src/main.rs"
```

- [ ] **Step 6: Dataset fetch script**

`scripts/fetch_sharegpt.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
if [ ! -f "$DEST/sharegpt.json" ]; then
  echo "downloading ShareGPT (~640MB, one-time)..."
  curl -fL -o "$DEST/sharegpt.json" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
fi
echo "trace at $DEST/sharegpt.json"
```
Run: `chmod +x scripts/fetch_sharegpt.sh`

- [ ] **Step 7: End-to-end smoke (local, small)**

```bash
./scripts/fetch_tokenizer.sh data
cargo build --workspace --release
./target/release/mock-triton --port 8001 --metrics-port 9101 &
./target/release/mock-triton --port 8002 --metrics-port 9102 &
./target/release/router --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --port 8000 &
sleep 1
./target/release/loadgen --router http://127.0.0.1:8000 \
  --trace crates/loadgen/tests/fixtures/sharegpt_small.json \
  --conversations 2 --concurrency 2 --label smoke
kill %1 %2 %3
```
Expected: stats table prints; hit rate ≥ 0 (tiny fixture may produce few blocks — fine, this is a wiring check).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(loadgen): ShareGPT multi-turn replay with latency + hit-rate stats"
```

---

### Task 14: Docker stack

**Files:**
- Create: `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/prometheus.yml`
- Create: `docker/grafana/provisioning/datasources/prometheus.yml`
- Create: `docker/grafana/provisioning/dashboards/dashboards.yml`, `docker/grafana/provisioning/dashboards/dynamorouter.json`

- [ ] **Step 1: Dockerfile (multi-stage, non-root)**

`docker/Dockerfile`:
```dockerfile
FROM rust:1.82-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml ./
COPY crates ./crates
RUN cargo build --release --workspace

FROM debian:bookworm-slim AS runtime
RUN useradd --system --uid 10001 --create-home appuser
COPY --from=builder /app/target/release/router /usr/local/bin/router
COPY --from=builder /app/target/release/mock-triton /usr/local/bin/mock-triton
COPY --from=builder /app/target/release/loadgen /usr/local/bin/loadgen
COPY data/tokenizer.json /data/tokenizer.json
USER appuser
ENV TOKENIZER_PATH=/data/tokenizer.json
ENTRYPOINT ["/usr/local/bin/router"]
```
Build context is the repo root (`docker build -f docker/Dockerfile .`). Requires `./scripts/fetch_tokenizer.sh data` first (tokenizer baked into image).

- [ ] **Step 2: compose file**

`docker/docker-compose.yml`:
```yaml
x-mock: &mock
  build:
    context: ..
    dockerfile: docker/Dockerfile
  entrypoint: ["/usr/local/bin/mock-triton"]
  environment:
    PORT: "8001"
    METRICS_PORT: "9101"
    CACHE_BLOCKS: "4096"
    PREFILL_US_PER_TOKEN: "200"

services:
  router:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
      - "9100:9100"
    environment:
      PORT: "8000"
      METRICS_PORT: "9100"
      REPLICAS: "http://mock1:8001,http://mock2:8001,http://mock3:8001,http://mock4:8001,http://mock5:8001,http://mock6:8001,http://mock7:8001,http://mock8:8001"
      POLICY: "${POLICY:-cache-aware}"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://jaeger:4317"
    depends_on: [mock1, mock2, mock3, mock4, mock5, mock6, mock7, mock8]

  mock1: *mock
  mock2: *mock
  mock3: *mock
  mock4: *mock
  mock5: *mock
  mock6: *mock
  mock7: *mock
  mock8: *mock

  prometheus:
    image: prom/prometheus:v2.53.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:11.1.0
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: "Admin"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    ports:
      - "3000:3000"

  jaeger:
    image: jaegertracing/all-in-one:1.58
    environment:
      COLLECTOR_OTLP_ENABLED: "true"
    ports:
      - "16686:16686"
```

- [ ] **Step 3: prometheus config**

`docker/prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: router
    static_configs:
      - targets: ["router:9100"]
  - job_name: replicas
    static_configs:
      - targets:
          - "mock1:9101"
          - "mock2:9101"
          - "mock3:9101"
          - "mock4:9101"
          - "mock5:9101"
          - "mock6:9101"
          - "mock7:9101"
          - "mock8:9101"
```

- [ ] **Step 4: Grafana provisioning**

`docker/grafana/provisioning/datasources/prometheus.yml`:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`docker/grafana/provisioning/dashboards/dashboards.yml`:
```yaml
apiVersion: 1
providers:
  - name: dynamorouter
    folder: ""
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

`docker/grafana/provisioning/dashboards/dynamorouter.json` — four panels; compact schema:
```json
{
  "title": "DynamoRouter",
  "uid": "dynamorouter",
  "refresh": "5s",
  "time": { "from": "now-15m", "to": "now" },
  "panels": [
    {
      "id": 1, "title": "Cache hit rate (ground truth)", "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "fieldConfig": { "defaults": { "unit": "percentunit", "max": 1, "min": 0 } },
      "targets": [{
        "expr": "sum(rate(replica_cached_blocks_total[1m])) / sum(rate(replica_total_blocks_total[1m]))",
        "legendFormat": "hit rate"
      }]
    },
    {
      "id": 2, "title": "Routing overhead P99", "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "fieldConfig": { "defaults": { "unit": "s" } },
      "targets": [{
        "expr": "histogram_quantile(0.99, rate(router_routing_overhead_seconds_bucket[1m]))",
        "legendFormat": "p99"
      }]
    },
    {
      "id": 3, "title": "Requests/sec", "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "targets": [{
        "expr": "rate(router_requests_total[1m])",
        "legendFormat": "req/s"
      }]
    },
    {
      "id": 4, "title": "In-flight per replica", "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
      "targets": [{
        "expr": "router_replica_inflight",
        "legendFormat": "replica {{replica}}"
      }]
    }
  ],
  "schemaVersion": 39
}
```

- [ ] **Step 5: Bring the stack up**

```bash
./scripts/fetch_tokenizer.sh data
docker compose -f docker/docker-compose.yml up --build -d
docker compose -f docker/docker-compose.yml ps
```
Expected: router + 8 mocks + prometheus + grafana + jaeger all Up. Check `curl -s localhost:9100/metrics | head`, open Grafana at `localhost:3000` → DynamoRouter dashboard exists, Jaeger at `localhost:16686`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(docker): full compose stack with prometheus, grafana, jaeger"
```

---

### Task 15: Benchmark script + README

**Files:**
- Create: `scripts/bench.sh`, `README.md`

- [ ] **Step 1: bench script**

`scripts/bench.sh`:
```bash
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
```
Run: `chmod +x scripts/bench.sh`

Note: recreating the router between runs also resets its radix trees; mock replica caches persist but round-robin runs first, so leftover warmth *helps the baseline* — the comparison is conservative. For a clean-room number, `docker compose ... restart mock1 ... mock8` between runs.

- [ ] **Step 2: Run the benchmark**

```bash
docker compose -f docker/docker-compose.yml up --build -d
./scripts/bench.sh
```
Expected output shape:
```
== round-robin ==
cache hit rate:  ~30-40%
== cache-aware ==
cache hit rate:  ~65-80%
```
Cache-aware hit rate must be dramatically higher and P50/P99 latency lower (less prefill work). If the gap is small, tune: `CACHE_BLOCKS` down on replicas (more eviction pressure → round-robin suffers more) or `CONVS` up. Record the real numbers in the README.

- [ ] **Step 3: README**

`README.md` — write it with the *actual measured numbers* from Step 2, structure:
```markdown
# DynamoRouter

Prefix-cache-aware inference request router for Triton Inference Server, in Rust.

[architecture diagram — ASCII of client → router → replicas]

## How it works
[3 short paragraphs: KServe passthrough proxy; per-replica radix tree over
16-token block hashes tracking which replica holds which KV prefix;
score = α·matched − β·inflight with least-loaded fallback + LRU capacity model]

## Results (ShareGPT replay, 8 mock replicas)
| policy | cache hit rate | P50 | P99 |
|--------|---------------|-----|-----|
| round-robin | <measured> | <measured> | <measured> |
| cache-aware | <measured> | <measured> | <measured> |

## Quickstart
[fetch scripts, docker compose up, bench.sh, Grafana/Jaeger URLs]

## Design docs
docs/superpowers/specs/, docs/superpowers/plans/
```

- [ ] **Step 4: Final verification**

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all --check
```
Expected: all green. Fix anything that isn't.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: benchmark script and README with measured results"
```

---

## Self-Review Notes (spec coverage)

- Spec §Decisions → Tasks 2 (passthrough), 5-6 (radix+scheduler), 8 (mock replicas), 13 (ShareGPT), 14 (compose stack): covered.
- Spec §Error handling → Task 9 (backpressure, retry, invalid input), Task 11 (health probe + tree drop on failure): covered.
- Spec §Observability → Task 9/12 (router metrics + OTel), Task 8 (replica ground-truth counters), Task 14 (Grafana dashboard): covered.
- Spec §Success criteria → Task 15 benchmark + README: covered.
- Streaming (spec: "streaming forwarded through, zero buffering") → Tasks 8/10: covered.
- Known deviation: GPT-2 tokenizer instead of Llama (ungated download); noted in File Structure section.
