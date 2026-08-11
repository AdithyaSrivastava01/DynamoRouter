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
                TextEncoder::new()
                    .encode_to_string(&registry.gather())
                    .unwrap_or_default()
            }
        }),
    );
    let metrics_addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.metrics_port));
    tokio::spawn(async move {
        // A replica whose metrics endpoint is down is unobservable to the
        // benchmark/compose stack; fail loudly instead of an unwrap panic
        // buried in a detached task (which would just silently vanish).
        let listener = match tokio::net::TcpListener::bind(metrics_addr).await {
            Ok(listener) => listener,
            Err(e) => {
                tracing::error!("failed to bind metrics listener on {metrics_addr}: {e}");
                std::process::exit(1);
            }
        };
        if let Err(e) = axum::serve(listener, metrics_app).await {
            tracing::error!("metrics server exited: {e}");
            std::process::exit(1);
        }
    });

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("mock-triton listening on {addr}");
    tonic::transport::Server::builder()
        .add_service(GrpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
