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
