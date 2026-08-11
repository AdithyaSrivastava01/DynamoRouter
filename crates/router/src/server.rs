#![allow(clippy::result_large_err)]

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

pub struct RouterInner {
    pub scheduler: Mutex<Scheduler>,
    pub clients: Vec<GrpcInferenceServiceClient<Channel>>,
    pub tokenizer: PromptTokenizer,
    pub metrics: Metrics,
}

impl RouterInner {
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
        self.metrics
            .routing_overhead_seconds
            .observe(start.elapsed().as_secs_f64());
        self.metrics.requests_total.inc();
        self.metrics
            .matched_blocks_total
            .inc_by(decision.matched_blocks as u64);
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

        // up to 2 attempts; transport errors mark the replica unhealthy
        let mut last_err = Status::unavailable("no attempt made");
        for _attempt in 0..2 {
            let decision = self.inner.route(&text)?;
            let mut client = self.inner.clients[decision.replica].clone();
            let result = client.model_infer(req.clone()).await;
            self.inner.complete(decision.replica);
            match result {
                Ok(resp) => return Ok(resp),
                Err(status) if status.code() == tonic::Code::Unavailable => {
                    self.inner.mark_unhealthy(decision.replica);
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
        request: Request<Streaming<pb::ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut client_rx = request.into_inner();
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<pb::ModelStreamInferResponse, Status>>(16);
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
}
