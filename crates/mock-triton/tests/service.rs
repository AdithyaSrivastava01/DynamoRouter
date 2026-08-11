use mock_triton::service::{MockConfig, MockTritonService};
use protocol::pb;
use protocol::{GrpcInferenceServiceClient, GrpcInferenceServiceServer};

const TOKENIZER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/tokenizer.json");

async fn spawn_service(cfg: MockConfig) -> (String, tokio::task::JoinHandle<()>) {
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

fn int64_param(resp: &pb::ModelInferResponse, name: &str) -> Option<i64> {
    match resp
        .parameters
        .get(name)
        .and_then(|p| p.parameter_choice.as_ref())
    {
        Some(pb::infer_parameter::ParameterChoice::Int64Param(v)) => Some(*v),
        _ => None,
    }
}

// Long repeated prefix so the prompt spans several 16-token blocks.
fn long_prompt(suffix: &str) -> String {
    format!(
        "{} {suffix}",
        "The quick brown fox jumps over the lazy dog. ".repeat(30)
    )
}

#[tokio::test]
async fn server_live_reports_true() {
    let cfg = MockConfig {
        tokenizer_path: TOKENIZER.into(),
        prefill_us_per_token: 0,
        ..Default::default()
    };
    let (url, _handle) = spawn_service(cfg).await;
    let mut client = GrpcInferenceServiceClient::connect(url).await.unwrap();

    let resp = client
        .server_live(pb::ServerLiveRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(resp.live);
}

#[tokio::test]
async fn model_infer_reports_cold_then_warm_cache() {
    let cfg = MockConfig {
        tokenizer_path: TOKENIZER.into(),
        prefill_us_per_token: 0,
        ..Default::default()
    };
    let (url, _handle) = spawn_service(cfg).await;
    let mut client = GrpcInferenceServiceClient::connect(url).await.unwrap();

    let first = client
        .model_infer(infer_request(&long_prompt("turn one")))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(int64_param(&first, "cached_blocks"), Some(0));
    let total = int64_param(&first, "total_blocks").unwrap();
    assert!(total > 1, "prompt should span multiple 16-token blocks");

    let second = client
        .model_infer(infer_request(&long_prompt(
            "turn one and then some more words",
        )))
        .await
        .unwrap()
        .into_inner();
    let cached_second = int64_param(&second, "cached_blocks").unwrap();
    assert!(
        cached_second > 0,
        "repeated prefix should hit the warm cache"
    );
}

#[tokio::test]
async fn model_stream_infer_emits_configured_chunk_count_with_params_on_last() {
    let cfg = MockConfig {
        tokenizer_path: TOKENIZER.into(),
        prefill_us_per_token: 0,
        decode_chunks: 3,
        decode_interval_ms: 1,
        ..Default::default()
    };
    let (url, _handle) = spawn_service(cfg).await;
    let mut client = GrpcInferenceServiceClient::connect(url).await.unwrap();

    let outbound = tokio_stream::once(infer_request(&long_prompt("stream me")));
    let response = client.model_stream_infer(outbound).await.unwrap();
    let mut inbound = response.into_inner();

    let mut chunks = Vec::new();
    while let Some(msg) = tokio_stream::StreamExt::next(&mut inbound).await {
        chunks.push(msg.unwrap());
    }

    assert_eq!(
        chunks.len(),
        3,
        "should emit exactly decode_chunks responses"
    );
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(chunk.error_message.is_empty());
        let infer_resp = chunk.infer_response.as_ref().unwrap();
        let is_last = i == chunks.len() - 1;
        if is_last {
            assert!(int64_param(infer_resp, "total_blocks").unwrap() > 0);
            assert_eq!(int64_param(infer_resp, "cached_blocks"), Some(0)); // cold prefix
        } else {
            // non-final chunks carry placeholder zeroed params
            assert_eq!(int64_param(infer_resp, "cached_blocks"), Some(0));
            assert_eq!(int64_param(infer_resp, "total_blocks"), Some(0));
        }
    }
}
