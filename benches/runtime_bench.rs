//! Manual-timing benchmarks (no criterion: it is unavailable in this
//! offline environment, so this harness reports ns/op directly).
//!
//! Run with `cargo bench`. Numbers are wall-clock on this machine and are
//! useful for relative comparison, not absolute claims.

use dynamic_llm_runtime::adapters::http::parse_sse_line;
use dynamic_llm_runtime::models::registry::ModelRegistry;
use dynamic_llm_runtime::{InferenceRequest, ModelRouter};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

fn bench_sync(name: &str, iters: u64, mut f: impl FnMut()) {
    for _ in 0..iters.min(1000) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    println!(
        "{name:>28}: {iters:>8} iters in {:>8.2}ms ({:>10.1} ns/op)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn sample_request() -> InferenceRequest {
    InferenceRequest {
        request_id: Some("bench".to_string()),
        prompt: "Summarize the quarterly report in three bullet points.".to_string(),
        model: None,
        max_tokens: Some(256),
        temperature: Some(0.7),
        top_p: None,
        stop: None,
        stream: Some(false),
        tools: None,
        metadata: None,
    }
}

fn main() {
    println!("=== dynamic-llm-runtime benchmarks ===");

    // 1. Registry lookup.
    let registry = ModelRegistry::new();
    bench_sync("registry::get", 200_000, || {
        let info = registry.get("llama-3.2-3b").unwrap();
        std::hint::black_box(info.context_limit);
    });

    // 2. JSON request/response round-trip.
    bench_sync("serde::roundtrip", 100_000, || {
        let request = sample_request();
        let text = serde_json::to_string(&request).unwrap();
        let back: InferenceRequest = serde_json::from_str(&text).unwrap();
        std::hint::black_box(back.prompt.len());
    });

    // 3. SSE line parsing.
    let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
    bench_sync("sse::parse_line", 200_000, || {
        std::hint::black_box(parse_sse_line(line));
    });

    // 4. Async routing (all iterations inside one runtime).
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let registry = Arc::new(RwLock::new(ModelRegistry::new()));
    let router = ModelRouter::new(registry);
    let iters = 20_000u64;
    runtime.block_on(async {
        for _ in 0..1000 {
            let id = router.route(&sample_request()).await.unwrap();
            std::hint::black_box(id);
        }
        let start = Instant::now();
        for _ in 0..iters {
            let id = router.route(&sample_request()).await.unwrap();
            std::hint::black_box(id);
        }
        let elapsed = start.elapsed();
        println!(
            "{:>28}: {iters:>8} iters in {:>8.2}ms ({:>10.1} ns/op)",
            "router::route",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / iters as f64,
        );
    });

    println!("=== done ===");
}
