use async_trait::async_trait;
use dynamic_llm_runtime::config::RuntimeConfig;
use dynamic_llm_runtime::core::error::Result;
use dynamic_llm_runtime::core::types::{InferenceRequest, InferenceResponse};
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::{ModelInfo, ModelRegistry};
use dynamic_llm_runtime::scheduling::predictor::NextModelPredictor;
use dynamic_llm_runtime::telemetry::Telemetry;
use dynamic_llm_runtime::{ModelRouter, RoutingStrategy, RoutingWeights};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

fn model(id: &str, affinity: &[&str]) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        name: id.to_string(),
        backend: "openai".to_string(),
        capabilities: ModelCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_structured_output: false,
            task_affinity: affinity.iter().map(|task| task.to_string()).collect(),
        },
        context_limit: 4096,
        memory_required_mb: 0,
        quantization: None,
        enabled: true,
        gguf_path: None,
        local_port: None,
        priority: 0,
    }
}

fn request() -> InferenceRequest {
    InferenceRequest {
        request_id: None,
        prompt: "hello".to_string(),
        model: None,
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
        stream: None,
        tools: None,
        metadata: None,
    }
}

#[tokio::test]
async fn adaptive_routing_orders_by_synthetic_telemetry() {
    let mut registry = ModelRegistry::new();
    registry.clear();
    registry.register(model("fast-good", &["general"]));
    registry.register(model("slow-bad", &["general"]));
    registry.register(model("unknown", &["general"]));
    let registry = Arc::new(RwLock::new(registry));

    let telemetry = Arc::new(Telemetry::new());
    for _ in 0..5 {
        telemetry.record_observation("fast-good", 50.0, true).await;
    }
    for _ in 0..5 {
        telemetry
            .record_observation("slow-bad", 2000.0, false)
            .await;
    }
    // "unknown" has no samples: neutral prior, between the two extremes.

    let router = ModelRouter::new(registry)
        .with_strategy(RoutingStrategy::Adaptive)
        .with_telemetry(telemetry)
        .with_weights(RoutingWeights::default());
    let candidates = router.route_candidates(&request()).await.unwrap();
    assert_eq!(candidates[0], "fast-good");
    assert_eq!(candidates[1], "unknown");
    assert_eq!(candidates[2], "slow-bad");

    // Quality-aware shares the scoring; task affinity adds a bonus.
    let mut registry = ModelRegistry::new();
    registry.clear();
    registry.register(model("generalist", &["general"]));
    registry.register(model("coder", &["code"]));
    let registry = Arc::new(RwLock::new(registry));
    let telemetry = Arc::new(Telemetry::new());
    let router = ModelRouter::new(registry)
        .with_strategy(RoutingStrategy::QualityAware)
        .with_telemetry(telemetry)
        .with_weights(RoutingWeights::default());
    let mut req = request();
    req.metadata = Some(HashMap::from([("task".to_string(), "code".to_string())]));
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(candidates[0], "coder");
}

#[test]
fn predictor_counts_hits() {
    let mut predictor = NextModelPredictor::new();
    assert_eq!(predictor.hit_rate(), None);
    for model in ["a", "b", "a", "b"] {
        predictor.observe(model);
    }
    // After "b", the most frequent successor is "a".
    let predicted = predictor.predict();
    assert_eq!(predicted.as_deref(), Some("a"));
    assert!(predictor.record_outcome(predicted.as_deref(), "a"));
    assert!(!predictor.record_outcome(predicted.as_deref(), "b"));
    assert_eq!(predictor.hits(), 1);
    assert_eq!(predictor.total_predictions(), 2);
    assert!((predictor.hit_rate().unwrap() - 0.5).abs() < 1e-9);
}

#[tokio::test]
async fn predictor_hit_rate_mirrors_into_telemetry() {
    let telemetry = Telemetry::new();
    let mut predictor = NextModelPredictor::new();
    predictor.observe("a");
    predictor.observe("b");
    let predicted = predictor.predict_after("a");
    let hit = predictor.record_outcome(predicted.as_deref(), "b");
    telemetry.record_prefetch_outcome(hit).await;
    assert_eq!(telemetry.prefetch_hit_rate().await, Some(1.0));
    let snapshot = telemetry.snapshot().await;
    assert_eq!(snapshot.prefetch_hits, 1);
    assert_eq!(snapshot.prefetch_total, 1);
    assert_eq!(snapshot.prefetch_hit_rate, Some(1.0));
}

struct ConcurrencyProbe {
    in_flight: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
    delay_ms: u64,
}

#[async_trait]
impl dynamic_llm_runtime::adapters::Adapter for ConcurrencyProbe {
    fn name(&self) -> &str {
        "probe"
    }

    fn backend(&self) -> &str {
        "probe"
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        Ok(InferenceResponse {
            request_id: request.request_id.clone().unwrap_or_default(),
            model: "probe".to_string(),
            content: String::new(),
            finish_reason: None,
            usage: None,
            tool_calls: None,
            metadata: None,
        })
    }

    async fn load(&self) -> Result<()> {
        let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(current, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[tokio::test]
async fn prefetch_bound_is_respected() {
    use dynamic_llm_runtime::models::manager::ModelManager;

    let registry = Arc::new(RwLock::new(ModelRegistry::new()));
    let telemetry = Arc::new(Telemetry::new());
    let manager = Arc::new(
        ModelManager::new(registry.clone(), telemetry)
            .with_local_budget_mb(1 << 30)
            .with_prefetch_concurrency(2),
    );
    assert_eq!(manager.prefetch_concurrency(), 2);

    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));
    for id in ["p1", "p2", "p3", "p4", "p5"] {
        registry.write().await.register(ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            backend: "probe".to_string(),
            capabilities: ModelCapabilities {
                supports_tools: false,
                supports_streaming: false,
                supports_structured_output: false,
                task_affinity: vec![],
            },
            context_limit: 1024,
            memory_required_mb: 0,
            quantization: None,
            enabled: true,
            gguf_path: None,
            local_port: None,
            priority: 0,
        });
        manager
            .register_adapter(
                id,
                Arc::new(ConcurrencyProbe {
                    in_flight: in_flight.clone(),
                    max_seen: max_seen.clone(),
                    delay_ms: 80,
                }),
            )
            .await;
    }

    let handles: Vec<_> = ["p1", "p2", "p3", "p4", "p5"]
        .iter()
        .map(|id| manager.spawn_prefetch(id.to_string()))
        .collect();
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    let max = max_seen.load(Ordering::SeqCst);
    assert!(
        max <= 2,
        "prefetch concurrency bound violated: saw {max} concurrent loads"
    );
    assert!(max >= 1, "no loads observed");
    // Cloud (0MB) models all fit; every prefetch landed.
    assert_eq!(manager.loaded_models().await.len(), 5);
    // Residency hooks tracked the loads.
    assert_eq!(manager.residency_snapshot().await.len(), 5);
}

#[test]
fn old_config_files_still_parse() {
    // A config written before Phase 7 has none of the new keys; serde
    // defaults must fill them so the file still loads with defaults.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("runtime.json"),
        r#"{
            "host": "127.0.0.1", "port": 8080, "request_timeout_secs": 120,
            "max_retries": 3, "retry_backoff_ms": 200,
            "max_concurrent_requests": 16, "max_fallback_models": 2,
            "local_model_budget_mb": 8192, "routing_strategy": "rule_based",
            "eviction_policy": "lru", "fixed_model": null,
            "openai_base_url": "https://api.openai.com/v1",
            "openai_default_model": "gpt-3.5-turbo", "openai_api_key": null,
            "llamacpp_base_url": "http://127.0.0.1:8081",
            "llamacpp_default_model": "llama-3.2-3b"
        }"#,
    )
    .unwrap();
    let config = RuntimeConfig::load(dir.path());
    assert_eq!(config.port, 8080);
    let defaults = RuntimeConfig::default();
    assert_eq!(
        config.routing_latency_weight,
        defaults.routing_latency_weight
    );
    assert_eq!(
        config.routing_success_weight,
        defaults.routing_success_weight
    );
    assert_eq!(
        config.routing_affinity_bonus,
        defaults.routing_affinity_bonus
    );
    assert_eq!(config.prefetch_concurrency, defaults.prefetch_concurrency);
    assert!(config.prefetch_warm_models.is_empty());
    assert!(!config.predictive_prefetch_enabled);
    assert_eq!(
        config.predictor_history_limit,
        defaults.predictor_history_limit
    );

    // New strategies parse.
    assert_eq!(
        RoutingStrategy::parse("adaptive"),
        RoutingStrategy::Adaptive
    );
    assert_eq!(
        RoutingStrategy::parse("quality_aware"),
        RoutingStrategy::QualityAware
    );
}

#[tokio::test]
async fn residency_hooks_track_load_and_evict() {
    use dynamic_llm_runtime::models::manager::ModelManager;

    let registry = Arc::new(RwLock::new(ModelRegistry::new()));
    let telemetry = Arc::new(Telemetry::new());
    let manager = ModelManager::new(registry, telemetry).with_local_budget_mb(4096);
    let probe = |delay| ConcurrencyProbe {
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_seen: Arc::new(AtomicUsize::new(0)),
        delay_ms: delay,
    };
    manager
        .register_adapter("gpt-3.5-turbo", Arc::new(probe(0)))
        .await;
    manager.ensure_loaded("gpt-3.5-turbo").await.unwrap();
    let residency = manager.residency_snapshot().await;
    assert_eq!(residency.len(), 1);
    assert!(residency[0].resident);
    manager.unload("gpt-3.5-turbo").await.unwrap();
    let residency = manager.residency_snapshot().await;
    assert!(!residency[0].resident);
}
