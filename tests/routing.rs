use dynamic_llm_runtime::core::types::{InferenceRequest, Tool};
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::{ModelInfo, ModelRegistry};
use dynamic_llm_runtime::{ModelRouter, RoutingStrategy};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

fn model(id: &str, tools: bool, streaming: bool, memory_mb: u64, affinity: &[&str]) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        name: id.to_string(),
        backend: "openai".to_string(),
        capabilities: ModelCapabilities {
            supports_tools: tools,
            supports_streaming: streaming,
            supports_structured_output: false,
            task_affinity: affinity.iter().map(|task| task.to_string()).collect(),
        },
        context_limit: 4096,
        memory_required_mb: memory_mb,
        quantization: None,
        enabled: true,
        gguf_path: None,
        local_port: None,
        priority: 0,
    }
}

fn registry() -> Arc<RwLock<ModelRegistry>> {
    let mut registry = ModelRegistry::new();
    registry.clear();
    registry.register(model("coder", true, true, 2048, &["code"]));
    registry.register(model("summarizer", false, true, 512, &["summarization"]));
    registry.register(model("giant", true, false, 16384, &["code", "reasoning"]));
    Arc::new(RwLock::new(registry))
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
async fn explicit_model_wins_and_is_validated() {
    let router = ModelRouter::new(registry());
    let mut req = request();
    req.model = Some("giant".to_string());
    assert_eq!(router.route(&req).await.unwrap(), "giant");

    req.model = Some("missing".to_string());
    assert!(router.route(&req).await.is_err());
}

#[tokio::test]
async fn tool_requests_filter_to_capable_models() {
    let router = ModelRouter::new(registry());
    let mut req = request();
    req.tools = Some(vec![Tool {
        name: "t".to_string(),
        description: "t".to_string(),
        parameters: serde_json::json!({}),
    }]);
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(candidates, vec!["coder".to_string(), "giant".to_string()]);
}

#[tokio::test]
async fn empty_tools_vec_does_not_require_tools() {
    let router = ModelRouter::new(registry());
    let mut req = request();
    req.tools = Some(Vec::new());
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(candidates.len(), 3);
}

#[tokio::test]
async fn stream_requests_filter_to_streaming_models() {
    let router = ModelRouter::new(registry());
    let mut req = request();
    req.stream = Some(true);
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(
        candidates,
        vec!["coder".to_string(), "summarizer".to_string()]
    );
}

#[tokio::test]
async fn task_affinity_narrows_rule_based_routing() {
    let router = ModelRouter::new(registry());
    let mut req = request();
    req.metadata = Some(HashMap::from([("task".to_string(), "code".to_string())]));
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(candidates, vec!["coder".to_string(), "giant".to_string()]);

    // Unknown tasks fall back to all capable models instead of failing.
    req.metadata = Some(HashMap::from([("task".to_string(), "poetry".to_string())]));
    let candidates = router.route_candidates(&req).await.unwrap();
    assert_eq!(candidates.len(), 3);
}

#[tokio::test]
async fn resource_aware_orders_by_footprint() {
    let router = ModelRouter::new(registry()).with_strategy(RoutingStrategy::ResourceAware);
    let candidates = router.route_candidates(&request()).await.unwrap();
    assert_eq!(
        candidates,
        vec![
            "summarizer".to_string(),
            "coder".to_string(),
            "giant".to_string()
        ]
    );
}

#[tokio::test]
async fn fixed_strategy_requires_a_model() {
    let router = ModelRouter::new(registry()).with_strategy(RoutingStrategy::Fixed);
    assert!(router.route(&request()).await.is_err());

    let router = ModelRouter::new(registry())
        .with_strategy(RoutingStrategy::Fixed)
        .with_fixed_model("summarizer");
    assert_eq!(router.route(&request()).await.unwrap(), "summarizer");
}

#[tokio::test]
async fn no_capable_model_is_a_routing_error() {
    let mut bare = ModelRegistry::new();
    bare.clear();
    bare.register(model("plain", false, false, 0, &[]));
    let router = ModelRouter::new(Arc::new(RwLock::new(bare)));
    let mut req = request();
    req.stream = Some(true);
    req.tools = Some(vec![Tool {
        name: "t".to_string(),
        description: "t".to_string(),
        parameters: serde_json::json!({}),
    }]);
    assert!(router.route(&req).await.is_err());
}
