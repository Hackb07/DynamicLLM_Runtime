mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use common::{start_mock_server, test_request, MockServer};
use dynamic_llm_runtime::api::create_router;
use dynamic_llm_runtime::config::RuntimeConfig;
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::ModelInfo;
use dynamic_llm_runtime::Runtime;
use std::sync::Arc;
use tower::ServiceExt;

fn mock_model(id: &str, backend: &str, streaming: bool, tools: bool) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        name: id.to_string(),
        backend: backend.to_string(),
        capabilities: ModelCapabilities {
            supports_tools: tools,
            supports_streaming: streaming,
            supports_structured_output: false,
            task_affinity: vec!["general".to_string()],
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

async fn test_runtime(server: &MockServer) -> Arc<Runtime> {
    let config = RuntimeConfig {
        openai_base_url: server.base_url.clone(),
        llamacpp_base_url: server.base_url.clone(),
        request_timeout_secs: 10,
        max_retries: 1,
        retry_backoff_ms: 5,
        ..RuntimeConfig::default()
    };
    let models = vec![mock_model("mock-gpt", "openai", true, true)];
    Arc::new(Runtime::with_config(config, models).await.unwrap())
}

async fn call(
    runtime: &Arc<Runtime>,
    method: Method,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, Vec<u8>, String) {
    let builder = Request::builder().method(method).uri(uri);
    let request = match body {
        Some(json) => builder
            .header("content-type", "application/json")
            .body(Body::from(json.to_string()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let response = create_router(runtime.clone())
        .oneshot(request)
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes, content_type)
}

#[tokio::test]
async fn health_and_models_endpoints() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;

    let (status, body, _) = call(&runtime, Method::GET, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");

    let (status, body, _) = call(&runtime, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json.as_array().unwrap().len(), 1);
    assert_eq!(json[0]["id"], "mock-gpt");

    server.shutdown().await;
}

#[tokio::test]
async fn infer_roundtrip_and_validation() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;

    let mut request = test_request("api hello");
    request.model = Some("mock-gpt".to_string());
    let (status, body, _) = call(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some(serde_json::to_value(&request).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["content"]
        .as_str()
        .unwrap()
        .contains("mock-echo: api hello"));
    assert_eq!(json["model"], "mock-gpt");

    // Empty prompt is rejected.
    let bad = test_request("   ");
    let (status, body, _) = call(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some(serde_json::to_value(&bad).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("error").is_some());

    // Unknown model is a 404.
    let mut missing = test_request("hi");
    missing.model = Some("nope".to_string());
    let (status, _, _) = call(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some(serde_json::to_value(&missing).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    server.shutdown().await;
}

#[tokio::test]
async fn infer_stream_returns_sse() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;

    let mut request = test_request("stream me");
    request.model = Some("mock-gpt".to_string());
    request.stream = Some(true);
    let (status, body, content_type) = call(
        &runtime,
        Method::POST,
        "/v1/infer/stream",
        Some(serde_json::to_value(&request).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("text/event-stream"),
        "{content_type}"
    );
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("data: Hello"), "{text}");
    assert!(text.contains("data:  world"), "{text}");
    assert!(text.contains("data: [DONE]"), "{text}");

    server.shutdown().await;
}

#[tokio::test]
async fn status_reports_runtime_state() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;

    let mut request = test_request("count me");
    request.model = Some("mock-gpt".to_string());
    let (status, _, _) = call(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some(serde_json::to_value(&request).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body, _) = call(&runtime, Method::GET, "/v1/status", None).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "running");
    assert_eq!(json["models"], 1);
    assert_eq!(json["telemetry"]["completed_requests"], 1);
    assert!(json.get("memory").is_some());
    assert!(json.get("loaded_models").is_some());

    server.shutdown().await;
}

async fn call_raw(
    runtime: &Arc<Runtime>,
    method: Method,
    uri: &str,
    content_type: Option<&str>,
    body: Vec<u8>,
) -> (StatusCode, Vec<u8>) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(content_type) = content_type {
        builder = builder.header("content-type", content_type);
    }
    let request = builder.body(Body::from(body)).unwrap();
    let response = create_router(runtime.clone())
        .oneshot(request)
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

#[tokio::test]
async fn error_shape_is_json_for_bad_paths_methods_and_bodies() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;

    // Unknown path.
    let (status, body) = call_raw(&runtime, Method::GET, "/nope", None, Vec::new()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "unknown endpoint");

    // Known path, wrong method.
    let (status, body) = call_raw(&runtime, Method::GET, "/v1/infer", None, Vec::new()).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "method not allowed");

    // Malformed JSON body.
    let (status, body) = call_raw(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some("application/json"),
        b"{not json".to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("error").is_some());

    // Wrong content type.
    let (status, body) = call_raw(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some("text/plain"),
        b"hello".to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("error").is_some());

    server.shutdown().await;
}

#[tokio::test]
async fn failover_moves_to_next_candidate() {
    let server = start_mock_server().await;
    let config = RuntimeConfig {
        openai_base_url: server.base_url.clone(),
        llamacpp_base_url: "http://127.0.0.1:1".to_string(),
        request_timeout_secs: 10,
        max_retries: 0,
        retry_backoff_ms: 5,
        max_fallback_models: 2,
        ..RuntimeConfig::default()
    };
    // llama model sorts first (id order) but its server is down; the
    // runtime must fail over to the working OpenAI-compatible model.
    let models = vec![
        mock_model("a-local", "llama.cpp", true, false),
        mock_model("z-cloud", "openai", true, false),
    ];
    let runtime = Arc::new(Runtime::with_config(config, models).await.unwrap());

    let (status, body, _) = call(
        &runtime,
        Method::POST,
        "/v1/infer",
        Some(serde_json::to_value(test_request("fail over")).unwrap()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model"], "z-cloud");

    server.shutdown().await;
}
