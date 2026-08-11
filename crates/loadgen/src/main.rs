use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use loadgen::trace::load_sharegpt;
use protocol::pb;
use protocol::GrpcInferenceServiceClient;
use tokio::sync::Semaphore;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    router: String,
    #[arg(long, default_value = "data/sharegpt.json")]
    trace: String,
    #[arg(long, default_value_t = 200)]
    conversations: usize,
    #[arg(long, default_value_t = 32)]
    concurrency: usize,
    #[arg(long, default_value = "run")]
    label: String,
}

struct RequestResult {
    latency_ms: f64,
    cached_blocks: i64,
    total_blocks: i64,
}

/// Outcome of replaying one conversation's turns. `failed` is set when a
/// turn errors and the conversation is abandoned (remaining turns never
/// attempted) — at most one failed request per conversation, since replay
/// stops at the first error rather than pressing on.
struct ConvOutcome {
    results: Vec<RequestResult>,
    failed: bool,
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

fn param_i64(resp: &pb::ModelInferResponse, key: &str) -> i64 {
    match resp
        .parameters
        .get(key)
        .and_then(|p| p.parameter_choice.as_ref())
    {
        Some(pb::infer_parameter::ParameterChoice::Int64Param(v)) => *v,
        _ => 0,
    }
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let convs = load_sharegpt(&args.trace, args.conversations)?;
    eprintln!("loaded {} conversations from {}", convs.len(), args.trace);

    let sem = Arc::new(Semaphore::new(args.concurrency));
    let client = GrpcInferenceServiceClient::connect(args.router.clone()).await?;
    let start = Instant::now();
    let mut handles = Vec::new();

    for conv in convs {
        let sem = Arc::clone(&sem);
        let mut client = client.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let mut results = Vec::new();
            let mut failed = false;
            for turn in &conv.turns {
                let t0 = Instant::now();
                match client.model_infer(infer_request(turn)).await {
                    Ok(resp) => {
                        let resp = resp.into_inner();
                        results.push(RequestResult {
                            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
                            cached_blocks: param_i64(&resp, "cached_blocks"),
                            total_blocks: param_i64(&resp, "total_blocks"),
                        });
                    }
                    Err(status) => {
                        eprintln!("conv {}: {status}", conv.id);
                        failed = true;
                        break; // abandon conversation on error
                    }
                }
            }
            ConvOutcome { results, failed }
        }));
    }

    let mut all = Vec::new();
    let mut failed_requests = 0u64;
    let mut abandoned_conversations = 0u64;
    for h in handles {
        let outcome = h.await?;
        if outcome.failed {
            failed_requests += 1;
            abandoned_conversations += 1;
        }
        all.extend(outcome.results);
    }
    let wall = start.elapsed().as_secs_f64();

    let mut latencies: Vec<f64> = all.iter().map(|r| r.latency_ms).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cached: i64 = all.iter().map(|r| r.cached_blocks).sum();
    let total: i64 = all.iter().map(|r| r.total_blocks).sum();
    let hit_rate = if total > 0 {
        100.0 * cached as f64 / total as f64
    } else {
        0.0
    };

    println!("== {} ==", args.label);
    println!("requests:        {}", all.len());
    println!("wall time:       {wall:.1}s");
    println!("throughput:      {:.0} req/s", all.len() as f64 / wall);
    println!("latency P50:     {:.1} ms", percentile(&latencies, 0.50));
    println!("latency P99:     {:.1} ms", percentile(&latencies, 0.99));
    println!("cache hit rate:  {hit_rate:.1}% ({cached}/{total} blocks)");
    // Always printed, even when both counts are 0, so a clean run is
    // provably clean rather than the absence of the line being ambiguous
    // between "no errors" and "errors weren't checked".
    println!(
        "errors:          {failed_requests} requests failed, {abandoned_conversations} conversations abandoned"
    );
    Ok(())
}
