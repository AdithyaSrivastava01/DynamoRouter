#![allow(clippy::result_large_err)]

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
        registry
            .register(Box::new(cached_blocks_total.clone()))
            .unwrap();
        registry
            .register(Box::new(total_blocks_total.clone()))
            .unwrap();
        Self {
            registry,
            cached_blocks_total,
            total_blocks_total,
        }
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

    /// Simulate prefill: tokenizes `text`, looks it up in `cache`, records
    /// metrics, and sleeps for time proportional to *uncached* tokens.
    /// Returns `(cached_blocks, total_blocks)`. Shared by the unary and
    /// streaming paths so a tokenizer error is reported identically by
    /// both instead of silently degrading into an empty-token request (as
    /// the streaming path used to do when this logic was duplicated).
    async fn simulate_prefill(
        tokenizer: &tokenizers::Tokenizer,
        cache: &Mutex<crate::cache::KvCacheSim>,
        metrics: &MockMetrics,
        cfg: &MockConfig,
        text: &str,
    ) -> Result<(usize, usize), Status> {
        let ids = tokenizer
            .encode(text, false)
            .map_err(|e| Status::invalid_argument(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        let hashes = block_hashes(&ids);
        let total = hashes.len();
        let cached = cache.lock().unwrap().lookup_insert(&hashes);
        metrics.cached_blocks_total.inc_by(cached as u64);
        metrics.total_blocks_total.inc_by(total as u64);
        let uncached_tokens = ids.len().saturating_sub(cached * BLOCK_SIZE);
        let delay = Duration::from_micros(uncached_tokens as u64 * cfg.prefill_us_per_token);
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
        let (cached, total) = Self::simulate_prefill(
            &self.tokenizer,
            &self.cache,
            &self.metrics,
            &self.cfg,
            &text,
        )
        .await?;
        Ok(Response::new(Self::make_response(
            &req,
            cached,
            total,
            "mock-completion",
        )))
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<pb::ModelStreamInferResponse, Status>> + Send>>;

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<pb::ModelStreamInferResponse, Status>>(16);
        let cfg = self.cfg.clone();
        let cache = Arc::clone(&self.cache);
        let tokenizer = Arc::clone(&self.tokenizer);
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            // The router opens one stream per request, so in practice this
            // loop processes exactly one request per stream; it stays a
            // loop (rather than a single read) to tolerate a client that
            // reuses the stream, which is served serially by design — the
            // next message isn't read until the current one's decode loop
            // finishes.
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
                let (cached, total) =
                    match Self::simulate_prefill(&tokenizer, &cache, &metrics, &cfg, &text).await {
                        Ok(v) => v,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Instant;

    const TOKENIZER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/tokenizer.json");

    fn load_tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_file(TOKENIZER).unwrap()
    }

    // Long enough to guarantee at least one full 16-token block regardless
    // of exact tokenization.
    fn long_prompt() -> String {
        "the quick brown fox jumps over the lazy dog ".repeat(3)
    }

    #[tokio::test(start_paused = true)]
    async fn cold_prefill_sleeps_proportional_to_uncached_tokens() {
        let tokenizer = load_tokenizer();
        let cfg = MockConfig {
            prefill_us_per_token: 1000,
            ..Default::default()
        };
        let cache = Mutex::new(crate::cache::KvCacheSim::new(cfg.cache_blocks));
        let metrics = MockMetrics::new();

        let text = long_prompt();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert!(!ids.is_empty(), "test prompt tokenized to nothing");

        let start = Instant::now();
        let (cached, total) =
            MockTritonService::simulate_prefill(&tokenizer, &cache, &metrics, &cfg, &text)
                .await
                .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(cached, 0); // cold: nothing was cached yet
        assert!(total >= 1, "prompt should span at least one full block");
        // Every token is uncached on a cold prefix, so the virtual sleep
        // must equal ids.len() * prefill_us_per_token exactly.
        assert_eq!(
            elapsed,
            Duration::from_micros(ids.len() as u64 * cfg.prefill_us_per_token)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn warm_prefill_only_sleeps_for_the_uncached_remainder() {
        let tokenizer = load_tokenizer();
        let cfg = MockConfig {
            prefill_us_per_token: 1000,
            ..Default::default()
        };
        let cache = Mutex::new(crate::cache::KvCacheSim::new(cfg.cache_blocks));
        let metrics = MockMetrics::new();

        let text = long_prompt();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let hashes = block_hashes(&ids);
        assert!(!hashes.is_empty(), "test prompt too short for a full block");
        // Pre-warm the cache directly (no simulated delay) so the first
        // simulate_prefill call below already finds a warm prefix.
        cache.lock().unwrap().lookup_insert(&hashes);

        let start = Instant::now();
        let (cached, total) =
            MockTritonService::simulate_prefill(&tokenizer, &cache, &metrics, &cfg, &text)
                .await
                .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(cached, total); // the whole hashed prefix is warm
                                   // Only the sub-block remainder (tokens past the last full block,
                                   // never hashed/cached) should incur delay.
        let remainder_tokens = ids.len() - total * BLOCK_SIZE;
        assert_eq!(
            elapsed,
            Duration::from_micros(remainder_tokens as u64 * cfg.prefill_us_per_token)
        );
    }
}
