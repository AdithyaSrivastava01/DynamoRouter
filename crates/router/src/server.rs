#![allow(clippy::result_large_err)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use prometheus::IntGauge;
use tokio_stream::Stream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};

use protocol::blocks::block_hashes;
use protocol::pb;
use protocol::{GrpcInferenceService, GrpcInferenceServiceClient};

use crate::scheduler::{PickError, RouteDecision, Scheduler};
use crate::telemetry::Metrics;
use crate::tokenizer::PromptTokenizer;

pub struct RouterInner {
    pub scheduler: Mutex<Scheduler>,
    // TODO(task 11): replica `Channel`s are built with plain `connect_lazy`
    // in main.rs; they should also get a send-timeout and keepalive via
    // `Endpoint` so a wedged (not dead) replica can't hang a request
    // indefinitely. Not wired up yet.
    pub clients: Vec<GrpcInferenceServiceClient<Channel>>,
    pub tokenizer: PromptTokenizer,
    pub metrics: Metrics,
    /// One inflight gauge per replica, pre-resolved once at construction
    /// so the hot path never calls `with_label_values` (a lock + string
    /// allocation) per request.
    pub replica_inflight_gauges: Vec<IntGauge>,
}

impl RouterInner {
    /// Build a `RouterInner`, pre-resolving one inflight gauge per
    /// replica from `clients.len()`.
    pub fn new(
        scheduler: Scheduler,
        clients: Vec<GrpcInferenceServiceClient<Channel>>,
        tokenizer: PromptTokenizer,
        metrics: Metrics,
    ) -> Self {
        let replica_inflight_gauges = (0..clients.len())
            .map(|i| {
                metrics
                    .replica_inflight
                    .with_label_values(&[&i.to_string()])
            })
            .collect();
        Self {
            scheduler: Mutex::new(scheduler),
            clients,
            tokenizer,
            metrics,
            replica_inflight_gauges,
        }
    }

    /// Prefer the tensor explicitly named `text_input`; fall back to the
    /// first input tensor for lenient smoke clients that don't set a name.
    fn extract_text(req: &pb::ModelInferRequest) -> Result<&str, Status> {
        let bytes = req
            .inputs
            .iter()
            .find(|i| i.name == "text_input")
            .or_else(|| req.inputs.first())
            .and_then(|i| i.contents.as_ref())
            .and_then(|c| c.bytes_contents.first())
            .ok_or_else(|| Status::invalid_argument("missing text_input bytes tensor"))?;
        std::str::from_utf8(bytes).map_err(|_| Status::invalid_argument("text_input not utf8"))
    }

    /// Tokenize + block-hash `text`. Runs exactly once per request —
    /// callers must not re-tokenize per retry attempt, since tokenization
    /// (not the scheduler lock) is the expensive part of routing overhead.
    /// An empty prompt tokenizes to an empty (cold) hash list rather than
    /// erroring; only an actual tokenizer failure becomes INVALID_ARGUMENT.
    fn hash_prompt(&self, text: &str) -> Result<Vec<u64>, Status> {
        let tokens = self
            .tokenizer
            .tokenize(text)
            .map_err(|e| Status::invalid_argument(format!("tokenize: {e}")))?;
        Ok(block_hashes(&tokens))
    }

    /// One routing decision under the scheduler mutex; observes routing
    /// overhead per decision (i.e. per attempt, including retries).
    ///
    /// Deliberately does NOT touch requests_total / matched_blocks_total /
    /// prompt_blocks_total: those count once per request, not once per
    /// attempt, so double-counting a retried request would skew them.
    /// Callers increment those via `record_served` using the decision that
    /// actually served the request.
    fn pick_replica(&self, hashes: &[u64]) -> Result<RouteDecision, Status> {
        let start = Instant::now();
        let picked = {
            let mut sched = self.scheduler.lock().unwrap();
            sched.pick(hashes)
        };
        self.metrics
            .routing_overhead_seconds
            .observe(start.elapsed().as_secs_f64());
        let decision = picked.map_err(|e| match e {
            PickError::NoHealthyReplicas => Status::unavailable("no healthy replicas"),
            PickError::AllSaturated => Status::resource_exhausted("all replicas saturated"),
        })?;
        self.replica_inflight_gauges[decision.replica].set(decision.inflight_after as i64);
        Ok(decision)
    }

    /// Record the three per-request counters exactly once, using the
    /// decision that actually served the request — the attempt whose
    /// response went back to the caller (unary), or the single pick made
    /// for a given message (streaming, which never retries).
    fn record_served(&self, prompt_blocks: usize, decision: &RouteDecision) {
        self.metrics.requests_total.inc();
        self.metrics
            .matched_blocks_total
            .inc_by(decision.matched_blocks as u64);
        self.metrics
            .prompt_blocks_total
            .inc_by(prompt_blocks as u64);
    }

    fn complete(&self, replica: usize) {
        let mut sched = self.scheduler.lock().unwrap();
        sched.complete(replica);
        let inflight = sched.inflight(replica) as i64;
        drop(sched);
        self.replica_inflight_gauges[replica].set(inflight);
    }

    fn mark_unhealthy(&self, replica: usize) {
        self.scheduler.lock().unwrap().mark_unhealthy(replica);
    }
}

#[derive(Clone)]
pub struct RouterService {
    pub inner: Arc<RouterInner>,
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
            let sched = self.inner.scheduler.lock().unwrap();
            (0..sched.num_replicas()).any(|i| sched.is_healthy(i))
        };
        Ok(Response::new(pb::ServerReadyResponse {
            ready: any_healthy,
        }))
    }

    async fn model_metadata(
        &self,
        req: Request<pb::ModelMetadataRequest>,
    ) -> Result<Response<pb::ModelMetadataResponse>, Status> {
        // forward to first healthy replica
        let idx = {
            let sched = self.inner.scheduler.lock().unwrap();
            (0..sched.num_replicas()).find(|&i| sched.is_healthy(i))
        }
        .ok_or_else(|| Status::unavailable("no healthy replicas"))?;
        let mut client = self.inner.clients[idx].clone();
        client.model_metadata(req.into_inner()).await
    }

    async fn model_infer(
        &self,
        request: Request<pb::ModelInferRequest>,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let req = request.into_inner();
        let text = RouterInner::extract_text(&req)?;
        let hashes = self.inner.hash_prompt(text)?;
        // `text` (and its borrow of `req`) isn't needed past this point.

        // Attempt 1.
        let decision = self.inner.pick_replica(&hashes)?;
        let mut client = self.inner.clients[decision.replica].clone();
        let result = client.model_infer(req.clone()).await;
        self.inner.complete(decision.replica);
        match result {
            Ok(resp) => {
                self.inner.record_served(hashes.len(), &decision);
                return Ok(resp);
            }
            Err(status) if is_transport_failure(&status) => {
                self.inner.mark_unhealthy(decision.replica);
            }
            Err(status) => return Err(status), // app-level error: propagate
        }

        // Attempt 2 (retry on another replica). This is the last attempt,
        // so `req` is moved instead of cloned.
        let decision = self.inner.pick_replica(&hashes)?;
        let mut client = self.inner.clients[decision.replica].clone();
        let result = client.model_infer(req).await;
        self.inner.complete(decision.replica);
        match result {
            Ok(resp) => {
                self.inner.record_served(hashes.len(), &decision);
                Ok(resp)
            }
            Err(status) if is_transport_failure(&status) => {
                self.inner.mark_unhealthy(decision.replica);
                Err(status)
            }
            Err(status) => Err(status), // app-level error: propagate
        }
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<pb::ModelStreamInferResponse, Status>> + Send>>;

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut client_rx = request.into_inner();
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<pb::ModelStreamInferResponse, Status>>(16);
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            use tokio_stream::StreamExt;
            // Requests on the client's stream are handled one at a time —
            // the next message isn't read until the current request's
            // upstream stream has ended or errored. No pipelining/overlap
            // across requests within a single client stream.
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
                                error_message: fmt_status(&status),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                let hashes = match inner.hash_prompt(text) {
                    Ok(h) => h,
                    Err(status) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: fmt_status(&status),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                // `text` (and its borrow of `req`) isn't needed past this
                // point, so `req` can be moved into the upstream call below.
                let decision = match inner.pick_replica(&hashes) {
                    Ok(d) => d,
                    Err(status) => {
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: fmt_status(&status),
                                infer_response: None,
                            }))
                            .await;
                        continue;
                    }
                };
                // Streaming never retries: the single pick made for this
                // message is also the one that (attempts to) serve it.
                inner.record_served(hashes.len(), &decision);

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
                                    // Mid-stream failure: no replay of
                                    // already-sent chunks (per spec) — just
                                    // report and move to the next request.
                                    let _ = tx
                                        .send(Ok(pb::ModelStreamInferResponse {
                                            error_message: format!(
                                                "upstream: {}",
                                                fmt_status(&status)
                                            ),
                                            infer_response: None,
                                        }))
                                        .await;
                                    break;
                                }
                            }
                        }
                    }
                    Err(status) => {
                        // Failed to even open the upstream stream: this is
                        // a transport-level failure like the unary path's,
                        // so mark the replica unhealthy the same way,
                        // before reporting it to the client.
                        if is_transport_failure(&status) {
                            inner.mark_unhealthy(decision.replica);
                        }
                        let _ = tx
                            .send(Ok(pb::ModelStreamInferResponse {
                                error_message: format!("upstream connect: {}", fmt_status(&status)),
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
}

/// Probe `ServerLive` on every replica every `interval`, flipping each
/// replica's health state in the scheduler and updating the
/// `healthy_replicas` gauge. Runs until the process exits.
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

fn fmt_status(status: &Status) -> String {
    format!("{}: {}", status.code(), status.message())
}

/// Whether `status` indicates the upstream connection/transport itself
/// failed (dead replica, reset connection, etc.) rather than an
/// application-level error from a live replica. Observed empirically: a
/// TCP endpoint that accepts then immediately closes the connection
/// surfaces to the client as `Cancelled`, not `Unavailable` — tonic/h2
/// report the stream as canceled when the connection resets before the
/// request completes. `Unknown` and `Internal` are included as other
/// transport-shaped failure modes seen from broken/misbehaving peers.
fn is_transport_failure(status: &Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::Unavailable
            | tonic::Code::Unknown
            | tonic::Code::Internal
            | tonic::Code::Cancelled
    )
}
