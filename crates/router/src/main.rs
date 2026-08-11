use std::sync::Arc;

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
    tracing_subscriber::fmt::init(); // Task 12 replaces this with router::telemetry::init_tracing()
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

    let inner = Arc::new(RouterInner::new(
        Scheduler::new(args.replicas.len(), policy, cfg),
        clients,
        PromptTokenizer::from_file(&args.tokenizer_path)?,
        Metrics::new(),
    ));

    spawn_health_prober(Arc::clone(&inner), std::time::Duration::from_secs(2));

    let registry = inner.metrics.registry.clone();
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
