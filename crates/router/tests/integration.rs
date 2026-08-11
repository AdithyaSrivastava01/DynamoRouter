use std::sync::Arc;
use std::sync::Mutex;

use protocol::pb;
use protocol::{GrpcInferenceServiceClient, GrpcInferenceServiceServer};
use router::scheduler::{Policy, Scheduler, SchedulerConfig};
use router::server::{RouterInner, RouterService};
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
        inner: Arc::new(RouterInner {
            scheduler: Mutex::new(Scheduler::new(replica_urls.len(), policy, cfg)),
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
