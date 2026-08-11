use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
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
    /// Per-request timeout for calls to a replica, in seconds. A request
    /// that exceeds this is a DeadlineExceeded, which the router
    /// deliberately does NOT retry (see `is_transport_failure` in
    /// server.rs) — it just fails the request rather than risking double
    /// compute on a replica that's merely slow.
    #[arg(long, env = "UPSTREAM_TIMEOUT_S", default_value_t = 30)]
    upstream_timeout_s: u64,
}

/// Resolves when either Ctrl+C (SIGINT) or SIGTERM is received, so the
/// server can drain in-flight requests before exiting under both a local
/// `Ctrl+C` and an orchestrator-issued SIGTERM (docker stop, k8s pod
/// termination, etc).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    router::telemetry::init_tracing();
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

    let upstream_timeout = Duration::from_secs(args.upstream_timeout_s);
    let clients: Vec<_> = args
        .replicas
        .iter()
        .map(|url| {
            let channel = Endpoint::from_shared(url.clone())
                .with_context(|| format!("bad replica url: {url}"))?
                .timeout(upstream_timeout)
                .connect_timeout(Duration::from_secs(5))
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .connect_lazy();
            anyhow::Ok(GrpcInferenceServiceClient::new(channel))
        })
        .collect::<anyhow::Result<_>>()?;

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
    // Bind before spawning: an in-use metrics port should fail router
    // startup loudly, not vanish as a silent panic inside a detached task.
    let metrics_listener = tokio::net::TcpListener::bind(metrics_addr)
        .await
        .with_context(|| format!("failed to bind metrics listener on {metrics_addr}"))?;
    tokio::spawn(async move {
        if let Err(e) = axum::serve(metrics_listener, metrics_app).await {
            tracing::error!("metrics server exited: {e}");
            std::process::exit(1);
        }
    });

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!(%addr, policy = args.policy, replicas = args.replicas.len(), "router up");
    tonic::transport::Server::builder()
        .add_service(GrpcInferenceServiceServer::new(RouterService { inner }))
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    // Flush any spans still buffered in the OTel batch exporter before the
    // process exits.
    router::telemetry::shutdown_tracing();
    Ok(())
}
