use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, IntGaugeVec, Opts, Registry};

pub struct Metrics {
    pub registry: Registry,
    pub requests_total: IntCounter,
    pub routing_overhead_seconds: Histogram,
    pub matched_blocks_total: IntCounter,
    pub prompt_blocks_total: IntCounter,
    pub replica_inflight: IntGaugeVec,
    pub healthy_replicas: IntGauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let requests_total = IntCounter::new("router_requests_total", "requests routed").unwrap();
        let routing_overhead_seconds = Histogram::with_opts(
            HistogramOpts::new("router_routing_overhead_seconds", "route decision time").buckets(
                vec![
                    0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.008, 0.01, 0.025,
                ],
            ),
        )
        .unwrap();
        let matched_blocks_total =
            IntCounter::new("router_matched_blocks_total", "predicted cache-hit blocks").unwrap();
        let prompt_blocks_total =
            IntCounter::new("router_prompt_blocks_total", "total prompt blocks").unwrap();
        let replica_inflight = IntGaugeVec::new(
            Opts::new("router_replica_inflight", "inflight per replica"),
            &["replica"],
        )
        .unwrap();
        let healthy_replicas =
            IntGauge::new("router_healthy_replicas", "healthy replica count").unwrap();
        for m in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(routing_overhead_seconds.clone()),
            Box::new(matched_blocks_total.clone()),
            Box::new(prompt_blocks_total.clone()),
            Box::new(replica_inflight.clone()),
            Box::new(healthy_replicas.clone()),
        ] {
            registry.register(m).unwrap();
        }
        Self {
            registry,
            requests_total,
            routing_overhead_seconds,
            matched_blocks_total,
            prompt_blocks_total,
            replica_inflight,
            healthy_replicas,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// fmt logging always; OTel layer only when `OTEL_EXPORTER_OTLP_ENDPOINT`
/// is set. Keeps tests/dev clean (no exporter, no network calls) while
/// letting the docker-compose stack export to Jaeger by setting the env
/// var.
pub fn init_tracing() {
    let fmt_layer = tracing_subscriber::fmt::layer();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        // opentelemetry 0.27 / opentelemetry-otlp 0.27 API: `with_endpoint`
        // lives on the `WithExportConfig` trait (must be imported), the
        // provider type is `trace::TracerProvider` (not `SdkTracerProvider`
        // — that name arrives in a later minor version), and
        // `with_batch_exporter` takes an explicit runtime handle since the
        // batch span processor needs somewhere to spawn its background
        // flush task.
        use opentelemetry_otlp::WithExportConfig;
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint.clone())
            .build()
            .unwrap_or_else(|e| {
                // This only fails on a malformed endpoint/config (e.g. bad
                // URI), not on the collector being unreachable — that
                // failure mode is async/lazy in the batch exporter and
                // wouldn't show up here. Panicking with the offending
                // endpoint value is a deliberate choice for a
                // configuration-time error: it's process startup, so
                // failing loudly with actionable detail beats limping
                // along with tracing silently disabled.
                panic!("failed to build OTLP exporter for endpoint {endpoint:?}: {e}")
            });
        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", "dynamorouter"),
            ]))
            .build();
        use opentelemetry::trace::TracerProvider as _;
        let tracer = provider.tracer("dynamorouter");
        opentelemetry::global::set_tracer_provider(provider);
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
    }
}

/// Flush and shut down the global OTel tracer provider, if one was
/// installed by [`init_tracing`]. Harmless no-op when OTel was never
/// initialized (the OTEL_EXPORTER_OTLP_ENDPOINT branch never ran). Call
/// this after the server has stopped accepting new requests but before
/// process exit, so any spans still buffered in the batch exporter get a
/// chance to ship instead of being dropped on the floor.
pub fn shutdown_tracing() {
    opentelemetry::global::shutdown_tracer_provider();
}
