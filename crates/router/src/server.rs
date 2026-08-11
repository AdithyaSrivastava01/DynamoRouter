#![allow(clippy::result_large_err)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use prometheus::IntGauge;
use tokio_stream::Stream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument;

use protocol::blocks::block_hashes;
use protocol::pb;
use protocol::{GrpcInferenceService, GrpcInferenceServiceClient};

use crate::scheduler::{PickError, RouteDecision, Scheduler};
use crate::telemetry::Metrics;
use crate::tokenizer::PromptTokenizer;

pub struct RouterInner {
    pub scheduler: Mutex<Scheduler>,
    /// gRPC clients for each replica. Channels are built in main.rs with
    /// `Endpoint::timeout` / `connect_timeout` / `tcp_keepalive` set, so a
    /// wedged (not dead) replica can't hang a request indefinitely.
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
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
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

impl RouterService {
    /// The unary routing/retry logic, run inside the `model_infer` parent
    /// span (see the trait method below) so the `route_decision` and
    /// `upstream_infer` child spans from both attempts land in one trace
    /// instead of two disconnected roots.
    async fn model_infer_attempts(
        &self,
        req: pb::ModelInferRequest,
    ) -> Result<Response<pb::ModelInferResponse>, Status> {
        let text = RouterInner::extract_text(&req)?;
        let hashes = self.inner.hash_prompt(text)?;
        // `text` (and its borrow of `req`) isn't needed past this point.

        // Attempt 1.
        let decision = {
            let span = tracing::info_span!("route_decision", attempt = 0);
            let _g = span.enter();
            self.inner.pick_replica(&hashes)?
        };
        let upstream_span = tracing::info_span!(
            "upstream_infer",
            replica = decision.replica,
            matched_blocks = decision.matched_blocks
        );
        let mut client = self.inner.clients[decision.replica].clone();
        let result = client
            .model_infer(req.clone())
            .instrument(upstream_span)
            .await;
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
        let decision = {
            let span = tracing::info_span!("route_decision", attempt = 1);
            let _g = span.enter();
            self.inner.pick_replica(&hashes)?
        };
        let upstream_span = tracing::info_span!(
            "upstream_infer",
            replica = decision.replica,
            matched_blocks = decision.matched_blocks
        );
        let mut client = self.inner.clients[decision.replica].clone();
        let result = client.model_infer(req).instrument(upstream_span).await;
        self.inner.complete(decision.replica);
        match result {
            Ok(resp) => {
                self.inner.record_served(hashes.len(), &decision);
                Ok(resp)
            }
            Err(status) if is_transport_failure(&status) => {
                self.inner.mark_unhealthy(decision.replica);
                // Both attempts exhausted: report a single unavailable
                // status summarizing the last failure rather than leaking
                // the second replica's raw status code, so callers see
                // "the router gave up" instead of an ambiguous per-replica
                // error.
                Err(Status::unavailable(format!(
                    "all attempts failed: {}: {}",
                    status.code(),
                    status.message()
                )))
            }
            Err(status) => Err(status), // app-level error: propagate
        }
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
        // Single parent span for the whole request so both attempts'
        // `route_decision` / `upstream_infer` children are one trace.
        let request_span = tracing::info_span!("model_infer", model = %req.model_name);
        self.model_infer_attempts(req)
            .instrument(request_span)
            .await
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<pb::ModelStreamInferResponse, Status>> + Send>>;

    // Deliberately unspanned: streaming requests are handled on a detached
    // `tokio::spawn`ed task decoupled from this call's tracing context, and
    // a single client stream can carry many independently-routed requests,
    // so there's no one natural parent span to attach child spans to here.
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
///
/// Replicas are probed concurrently (one spawned task per replica, each
/// with its own 1s timeout) so total detection latency for a pass stays
/// flat as replica count grows, instead of serializing up to
/// `n * 1s` worst-case if probes ran one after another. Scheduler updates
/// are applied afterwards under a single lock acquisition per pass rather
/// than one per replica.
pub fn spawn_health_prober(
    inner: Arc<RouterInner>,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Default (Burst) behavior fires back-to-back catch-up ticks after
        // a slow pass (e.g. the process was stalled or a probe took a
        // while); Delay just resumes on the normal cadence from whenever
        // the tick actually fires, which is what we want for a periodic
        // health check.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let n = inner.clients.len();

            let mut probes = tokio::task::JoinSet::new();
            for i in 0..n {
                let mut client = inner.clients[i].clone();
                probes.spawn(async move {
                    let live = matches!(
                        tokio::time::timeout(
                            std::time::Duration::from_secs(1),
                            client.server_live(pb::ServerLiveRequest {}),
                        )
                        .await,
                        Ok(Ok(resp)) if resp.get_ref().live
                    );
                    (i, live)
                });
            }
            let mut results = Vec::with_capacity(n);
            while let Some(joined) = probes.join_next().await {
                if let Ok(pair) = joined {
                    results.push(pair);
                }
                // A `JoinError` here means the probe task panicked; skip it
                // for this pass rather than poisoning the whole prober —
                // it'll be probed again next tick.
            }

            let mut healthy_count = 0i64;
            {
                let mut sched = inner.scheduler.lock().unwrap();
                for (i, live) in results {
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
///
/// `DeadlineExceeded` is deliberately NOT included here. The replica
/// `Endpoint`s built in main.rs carry a per-request `timeout`; when that
/// elapses the replica may well be alive and still doing real work on the
/// request. Treating a deadline expiry as a transport failure and
/// retrying it on another replica would risk *doubling* the compute cost
/// of an already-expensive request instead of recovering from a dead
/// peer, so it's surfaced to the caller as-is rather than triggering a
/// retry.
///
/// Retry-safety caveat: every code matched below can also occur *after*
/// the upstream has already started processing the request (e.g. the
/// response was in flight when the connection reset), so a retry here can
/// mean the same request is executed twice by two different replicas.
/// That's an accepted tradeoff for this mock/demo router; a production
/// router fronting a stateful backend would want idempotency keys on the
/// request so a duplicate delivery is safe to detect and drop.
fn is_transport_failure(status: &Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::Unavailable
            | tonic::Code::Unknown
            | tonic::Code::Internal
            | tonic::Code::Cancelled
    )
}
