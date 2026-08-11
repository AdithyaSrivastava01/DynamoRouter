use std::sync::Arc;

use protocol::pb;
use protocol::{GrpcInferenceServiceClient, GrpcInferenceServiceServer};
use router::scheduler::{Policy, Scheduler, SchedulerConfig};
use router::server::{spawn_health_prober, RouterInner, RouterService};
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
    spawn_router_with_cfg(replica_urls, policy, SchedulerConfig::default()).await
}

async fn spawn_router_with_cap(
    replica_urls: &[String],
    max_inflight: usize,
) -> (String, tokio::task::JoinHandle<()>) {
    let cfg = SchedulerConfig {
        max_inflight,
        ..Default::default()
    };
    spawn_router_with_cfg(replica_urls, Policy::CacheAware, cfg).await
}

async fn spawn_router_with_cfg(
    replica_urls: &[String],
    policy: Policy,
    cfg: SchedulerConfig,
) -> (String, tokio::task::JoinHandle<()>) {
    let mut clients = Vec::new();
    for url in replica_urls {
        let channel = tonic::transport::Endpoint::from_shared(url.clone())
            .unwrap()
            .connect_lazy();
        clients.push(GrpcInferenceServiceClient::new(channel));
    }
    let svc = RouterService {
        inner: Arc::new(RouterInner::new(
            Scheduler::new(replica_urls.len(), policy, cfg),
            clients,
            PromptTokenizer::from_file(TOKENIZER).unwrap(),
            Metrics::new(),
        )),
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
    match resp
        .parameters
        .get("cached_blocks")
        .and_then(|p| p.parameter_choice.as_ref())
    {
        Some(pb::infer_parameter::ParameterChoice::Int64Param(v)) => *v,
        _ => panic!("missing cached_blocks param"),
    }
}

// Long repeated prefix so prompts span many 16-token blocks.
fn long_prompt(suffix: &str) -> String {
    format!(
        "{} {suffix}",
        "The quick brown fox jumps over the lazy dog. ".repeat(30)
    )
}

#[tokio::test]
async fn same_prefix_requests_hit_cache_via_pinning() {
    let (m1, _h1) = spawn_mock(0).await;
    let (m2, _h2) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router(&[m1, m2], Policy::CacheAware).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url)
        .await
        .unwrap();

    let first = client
        .model_infer(infer_request(&long_prompt("turn one")))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(cached_blocks(&first), 0); // cold

    let second = client
        .model_infer(infer_request(&long_prompt(
            "turn one and then some more words",
        )))
        .await
        .unwrap()
        .into_inner();
    assert!(
        cached_blocks(&second) > 0,
        "pinned replica should have warm prefix"
    );
}

#[tokio::test]
async fn saturated_replicas_reject_fast() {
    let (m1, _h1) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router_with_cap(&[m1], 0).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url)
        .await
        .unwrap();
    let err = client
        .model_infer(infer_request("hello"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::ResourceExhausted);
}

#[tokio::test]
async fn streaming_forwards_chunks() {
    let (m1, _h1) = spawn_mock(0).await;
    let (router_url, _rh) = spawn_router(&[m1], Policy::CacheAware).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url)
        .await
        .unwrap();

    let outbound = tokio_stream::iter(vec![infer_request(&long_prompt("stream me"))]);
    let mut inbound = client
        .model_stream_infer(outbound)
        .await
        .unwrap()
        .into_inner();

    let mut chunks = Vec::new();
    while let Some(msg) = inbound.message().await.unwrap() {
        assert!(
            msg.error_message.is_empty(),
            "unexpected: {}",
            msg.error_message
        );
        chunks.push(msg.infer_response.unwrap());
    }
    assert_eq!(chunks.len(), MockConfig::default().decode_chunks);
    assert!(chunks
        .last()
        .unwrap()
        .parameters
        .contains_key("cached_blocks"));
}

#[tokio::test]
async fn failover_retries_on_dead_replica() {
    // one real mock + one dead endpoint that accepts TCP then closes
    let (m1, _h1) = spawn_mock(0).await;
    let dead_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", dead_listener.local_addr().unwrap());
    let dead_handle = tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = dead_listener.accept().await else {
                break;
            };
            drop(sock); // slam the door: transport error upstream
        }
    });

    // spawn_router already uses connect_lazy, so startup doesn't fail on
    // the dead endpoint.
    let (router_url, _rh) = spawn_router(&[dead_url, m1], Policy::RoundRobin).await;
    let mut client = GrpcInferenceServiceClient::connect(router_url)
        .await
        .unwrap();

    // RoundRobin starts at replica 0 (dead) -> transport error -> retry
    // lands on replica 1.
    let resp = client
        .model_infer(infer_request(&long_prompt("failover")))
        .await;
    assert!(
        resp.is_ok(),
        "retry should succeed on healthy replica: {resp:?}"
    );
    dead_handle.abort();
}

#[tokio::test]
async fn health_prober_marks_dead_replica_unhealthy() {
    // Direct RouterInner + spawn_health_prober test, no router HTTP server
    // needed: cheap because it exercises the prober loop against a real
    // dead TCP endpoint and a real mock, without going through gRPC twice.
    let (m1, _h1) = spawn_mock(0).await;
    let dead_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", dead_listener.local_addr().unwrap());
    let dead_handle = tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = dead_listener.accept().await else {
                break;
            };
            drop(sock);
        }
    });

    let clients: Vec<_> = [dead_url, m1]
        .iter()
        .map(|url| {
            let channel = tonic::transport::Endpoint::from_shared(url.clone())
                .unwrap()
                .connect_lazy();
            GrpcInferenceServiceClient::new(channel)
        })
        .collect();
    let inner = Arc::new(RouterInner::new(
        Scheduler::new(2, Policy::RoundRobin, SchedulerConfig::default()),
        clients,
        PromptTokenizer::from_file(TOKENIZER).unwrap(),
        Metrics::new(),
    ));

    let prober = spawn_health_prober(Arc::clone(&inner), std::time::Duration::from_millis(50));
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    prober.abort();
    dead_handle.abort();

    assert!(
        !inner.scheduler.lock().unwrap().is_healthy(0),
        "dead replica should be marked unhealthy"
    );
    assert!(
        inner.scheduler.lock().unwrap().is_healthy(1),
        "live replica should stay healthy"
    );
    assert_eq!(inner.metrics.healthy_replicas.get(), 1);
}
