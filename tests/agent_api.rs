//! HTTP-level tests for `POST /v1/agent/run` and `GET /v1/agent/tools`.
//!
//! Uses a dedicated agent-aware mock backend (tool calls with valid args,
//! scripted denial/loop/slow behaviors) so the shared `common` mock keeps
//! its exact behavior for the other suites.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use dynamic_llm_runtime::agents::{AgentRequest, AgentResponse};
use dynamic_llm_runtime::api::create_router;
use dynamic_llm_runtime::config::RuntimeConfig;
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::ModelInfo;
use dynamic_llm_runtime::Runtime;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tower::ServiceExt;

struct AgentMock {
    base_url: String,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl AgentMock {
    async fn shutdown(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for AgentMock {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn start_agent_mock() -> AgentMock {
    let app = Router::new()
        .route("/chat/completions", post(chat))
        .route("/v1/chat/completions", post(chat));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });
    AgentMock {
        base_url,
        shutdown: Some(shutdown_tx),
        handle: Some(handle),
    }
}

fn tool_message(name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": name,
                "arguments": serde_json::to_string(&args).unwrap(),
            }
        }]
    })
}

fn direct_message(prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": format!("mock-echo: {prompt}")
    })
}

fn final_message() -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": "agent final answer"
    })
}

async fn chat(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let model = body
        .get("model")
        .and_then(|model| model.as_str())
        .unwrap_or("mock-gpt")
        .to_string();
    let prompt = body
        .get("messages")
        .and_then(|messages| messages.get(0))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string();
    let tools = body
        .get("tools")
        .and_then(|tools| tools.as_array())
        .cloned()
        .unwrap_or_default();
    let first_tool = tools
        .first()
        .and_then(|tool| tool.get("function"))
        .and_then(|function| function.get("name"))
        .and_then(|name| name.as_str())
        .unwrap_or("tool")
        .to_string();

    // Scripted behaviors are selected by prompt markers from the tests.
    if prompt.contains("AGENT_SLOW") {
        tokio::time::sleep(Duration::from_secs(10)).await;
        return reply(&model, direct_message(&prompt), "stop", &prompt);
    }
    let (message, finish_reason) = if !tools.is_empty() && prompt.contains("AGENT_LOOP") {
        (
            tool_message(&first_tool, args_for(&first_tool)),
            "tool_calls",
        )
    } else if !tools.is_empty() && prompt.contains("AGENT_SNEAKY") {
        // Request a tool outside the test's allowlist on purpose.
        (
            tool_message("calc", serde_json::json!({"expression": "1+1"})),
            "tool_calls",
        )
    } else if !tools.is_empty() && prompt.contains("[tool ") {
        (final_message(), "stop")
    } else if !tools.is_empty() {
        (
            tool_message(&first_tool, args_for(&first_tool)),
            "tool_calls",
        )
    } else {
        (direct_message(&prompt), "stop")
    };
    reply(&model, message, finish_reason, &prompt)
}

fn args_for(tool: &str) -> serde_json::Value {
    match tool {
        "echo" => serde_json::json!({"text": "hello from mock"}),
        "calc" => serde_json::json!({"expression": "2+3"}),
        "read_file" => serde_json::json!({"path": "nope.txt"}),
        _ => serde_json::json!({}),
    }
}

fn reply(
    model: &str,
    message: serde_json::Value,
    finish_reason: &str,
    prompt: &str,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "id": "chatcmpl-agent-mock",
        "model": model,
        "choices": [{ "message": message, "finish_reason": finish_reason }],
        "usage": {
            "prompt_tokens": prompt.len() as u32 / 4,
            "completion_tokens": 8,
            "total_tokens": prompt.len() as u32 / 4 + 8
        }
    }))
}

fn mock_model() -> ModelInfo {
    ModelInfo {
        id: "mock-gpt".to_string(),
        name: "mock-gpt".to_string(),
        backend: "openai".to_string(),
        capabilities: ModelCapabilities {
            supports_tools: true,
            supports_streaming: true,
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

async fn test_runtime(mock: &AgentMock) -> Arc<Runtime> {
    let config = RuntimeConfig {
        openai_base_url: mock.base_url.clone(),
        llamacpp_base_url: mock.base_url.clone(),
        // High enough that agent-level timeouts fire first in timeout tests.
        request_timeout_secs: 30,
        max_retries: 1,
        retry_backoff_ms: 5,
        ..RuntimeConfig::default()
    };
    Arc::new(
        Runtime::with_config(config, vec![mock_model()])
            .await
            .unwrap(),
    )
}

async fn call(
    runtime: &Arc<Runtime>,
    method: Method,
    uri: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, Vec<u8>) {
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
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

fn agent_body(prompt: &str, tools: &[&str]) -> serde_json::Value {
    let request = AgentRequest::new(prompt)
        .with_allowed_tools(tools.iter().map(|tool| tool.to_string()).collect());
    serde_json::to_value(&request).unwrap()
}

#[tokio::test]
async fn agent_run_direct_answer_and_telemetry() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(agent_body("api hello", &[])),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: AgentResponse = serde_json::from_slice(&body).unwrap();
    assert!(response.is_success());
    assert!(response.answer.contains("mock-echo: api hello"));
    assert_eq!(response.steps_used, 1);
    assert!(!response.audit.is_empty());
    assert!(response
        .transcript
        .iter()
        .any(|line| line.contains("api hello")));

    // The agent run itself is recorded, alongside its inner inference call.
    let (status, body) = call(&runtime, Method::GET, "/v1/status", None).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["telemetry"]["requests_per_model"]["agent"], 1);
    assert!(json["agent"]["tools"].as_u64().unwrap() >= 4);

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_echo_tool() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(agent_body("echo something", &["echo"])),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: AgentResponse = serde_json::from_slice(&body).unwrap();
    assert!(response.is_success());
    assert_eq!(response.answer, "agent final answer");
    assert_eq!(response.steps_used, 2);
    let tool_event = response
        .audit
        .iter()
        .find(|event| event.kind == "tool_call")
        .expect("tool_call audit event");
    assert_eq!(tool_event.tool.as_deref(), Some("echo"));
    assert!(tool_event.success);
    assert!(response
        .transcript
        .iter()
        .any(|line| line.contains("tool echo result")));

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_calc_tool() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(agent_body("compute 2+3", &["calc"])),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: AgentResponse = serde_json::from_slice(&body).unwrap();
    assert!(response.is_success());
    assert_eq!(response.steps_used, 2);
    let tool_event = response
        .audit
        .iter()
        .find(|event| event.kind == "tool_call")
        .expect("tool_call audit event");
    assert_eq!(tool_event.tool.as_deref(), Some("calc"));
    assert!(tool_event.success);
    assert!(response
        .transcript
        .iter()
        .any(|line| line.contains("calc") && line.contains('5')));

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_disallowed_tool_denied() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    // The mock requests `calc` but only `echo` is allowlisted.
    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(agent_body("AGENT_SNEAKY please", &["echo"])),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let response: AgentResponse = serde_json::from_slice(&body).unwrap();
    assert!(!response.is_success());
    assert!(response.error.as_deref().unwrap().contains("denied"));
    assert!(response.answer.is_empty());
    // The failed tool call is still audited.
    let tool_event = response
        .audit
        .iter()
        .find(|event| event.kind == "tool_call")
        .expect("denial audit event");
    assert_eq!(tool_event.tool.as_deref(), Some("calc"));
    assert!(!tool_event.success);

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_max_steps_enforced() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let mut body = agent_body("AGENT_LOOP tick", &["now"]);
    body["max_steps"] = serde_json::json!(2);
    let (status, bytes) = call(&runtime, Method::POST, "/v1/agent/run", Some(body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let response: AgentResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!response.is_success());
    assert_eq!(response.steps_used, 2);
    assert!(response.error.as_deref().unwrap().contains("budget"));

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_timeout() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let mut body = agent_body("AGENT_SLOW hello", &[]);
    body["timeout_secs"] = serde_json::json!(1);
    let (status, bytes) = call(&runtime, Method::POST, "/v1/agent/run", Some(body)).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    let response: AgentResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!response.is_success());
    assert!(response.error.as_deref().unwrap().contains("timeout"));

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_rejects_empty_prompt() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(agent_body("   ", &[])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("error").is_some());

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_rejects_zero_limits() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    for (field, value) in [
        ("max_steps", serde_json::json!(0)),
        ("timeout_secs", serde_json::json!(0)),
        ("token_budget", serde_json::json!(0)),
        ("max_tokens_per_step", serde_json::json!(0)),
    ] {
        let mut body = agent_body("do it", &[]);
        body[field] = value;
        let (status, bytes) = call(&runtime, Method::POST, "/v1/agent/run", Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "field: {field}");
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.get("error").is_some(), "field: {field}");
    }

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_run_accepts_task_alias() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    // `task` aliases `prompt`; limits fall back to defaults when omitted.
    let (status, body) = call(
        &runtime,
        Method::POST,
        "/v1/agent/run",
        Some(serde_json::json!({"task": "alias hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: AgentResponse = serde_json::from_slice(&body).unwrap();
    assert!(response.is_success());
    assert!(response.answer.contains("alias hello"));

    mock.shutdown().await;
}

#[tokio::test]
async fn agent_tools_listed() {
    let mock = start_agent_mock().await;
    let runtime = test_runtime(&mock).await;

    let (status, body) = call(&runtime, Method::GET, "/v1/agent/tools", None).await;
    assert_eq!(status, StatusCode::OK);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let tools = json.as_array().unwrap();
    assert_eq!(tools.len(), 4);
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|name| name.as_str()))
        .collect();
    for expected in ["now", "echo", "calc", "read_file"] {
        assert!(names.contains(&expected), "missing tool: {expected}");
    }

    mock.shutdown().await;
}
