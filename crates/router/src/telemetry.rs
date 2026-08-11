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
