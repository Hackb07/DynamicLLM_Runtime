//! Phase 8: multi-agent sessions + distributed-inference stub.
//!
//! Sandbox-safe tests (no socket bind) run anywhere; tests that start the
//! shared mock backend need loopback and run in the escalated pass like
//! the other mock-server suites.

mod common;

use common::{start_mock_server, test_request, MockServer};
use dynamic_llm_runtime::agents::{AgentRequest, AgentState};
use dynamic_llm_runtime::config::{RuntimeConfig, WorkerConfig};
use dynamic_llm_runtime::core::error::Error;
use dynamic_llm_runtime::core::types::InferenceResponse;
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::ModelInfo;
use dynamic_llm_runtime::scheduling::workers::{WorkerInfo, WorkerRegistry};
use dynamic_llm_runtime::telemetry::Telemetry;
use dynamic_llm_runtime::Runtime;
use std::sync::atomic::Ordering;
use std::sync::Arc;

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

fn worker(id: &str, addr: &str, models: &[&str]) -> WorkerConfig {
    WorkerConfig {
        id: id.to_string(),
        addr: addr.to_string(),
        models: models.iter().map(|model| model.to_string()).collect(),
        capacity: 4,
    }
}

fn response(content: &str) -> InferenceResponse {
    InferenceResponse {
        request_id: "r".to_string(),
        model: "mock-gpt".to_string(),
        content: content.to_string(),
        finish_reason: Some("stop".to_string()),
        usage: None,
        tool_calls: None,
        metadata: None,
    }
}

#[tokio::test]
async fn workers_default_to_empty_single_node() {
    assert!(RuntimeConfig::default().workers.is_empty());
    // Old configs without the `workers` key still parse (serde default).
    let config: RuntimeConfig = serde_json::from_str(
        r#"{"host":"127.0.0.1","port":8080,"request_timeout_secs":120,
        "max_retries":3,"retry_backoff_ms":200,"max_concurrent_requests":16,
        "max_fallback_models":2,"local_model_budget_mb":8192,
        "routing_strategy":"rule_based","eviction_policy":"lru",
        "openai_base_url":"https://api.openai.com/v1",
        "openai_default_model":"gpt-3.5-turbo",
        "llamacpp_base_url":"http://127.0.0.1:8081",
        "llamacpp_default_model":"llama-3.2-3b"}"#,
    )
    .unwrap();
    assert!(config.workers.is_empty());

    let runtime = Runtime::with_config(RuntimeConfig::default(), vec![mock_model()])
        .await
        .unwrap();
    let status = runtime.get_status().await;
    assert_eq!(status["distributed"]["enabled"], false);
    assert_eq!(
        status["distributed"]["workers"].as_array().unwrap().len(),
        0
    );
    assert_eq!(status["agent"]["sessions"], 0);
}

#[tokio::test]
async fn worker_config_has_sane_insertion_defaults() {
    let worker: WorkerConfig =
        serde_json::from_str(r#"{"id":"w0","addr":"http://127.0.0.1:9090"}"#).unwrap();
    assert!(worker.models.is_empty());
    assert_eq!(worker.capacity, 4);
}

#[tokio::test]
async fn placement_orders_affinity_then_least_loaded() {
    let registry = WorkerRegistry::new(vec![
        WorkerInfo {
            id: "generalist".to_string(),
            addr: "http://g:8080".to_string(),
            models: Vec::new(),
            capacity: 10,
        },
        WorkerInfo {
            id: "busy-match".to_string(),
            addr: "http://b:8080".to_string(),
            models: vec!["m1".to_string()],
            capacity: 10,
        },
        WorkerInfo {
            id: "idle-match".to_string(),
            addr: "http://i:8080".to_string(),
            models: vec!["m1".to_string()],
            capacity: 10,
        },
    ]);
    let _load: Vec<_> = (0..9).map(|_| registry.acquire("busy-match")).collect();

    let ids: Vec<String> = registry
        .candidates(Some("m1"))
        .iter()
        .map(|worker| worker.id.clone())
        .collect();
    assert_eq!(ids, vec!["idle-match", "busy-match", "generalist"]);
}

#[tokio::test]
async fn worker_failure_falls_back_via_mock_client() {
    let registry = WorkerRegistry::new(vec![
        WorkerInfo {
            id: "bad".to_string(),
            addr: "http://bad:8080".to_string(),
            models: vec!["m1".to_string()],
            capacity: 4,
        },
        WorkerInfo {
            id: "good".to_string(),
            addr: "http://good:8080".to_string(),
            models: vec!["m1".to_string()],
            capacity: 4,
        },
    ]);
    let telemetry = Telemetry::new();

    // First worker fails, second succeeds: caller sees the remote answer
    // plus a telemetry note for the failure.
    let result = registry
        .try_remote(&telemetry, "m1", |worker| async move {
            if worker.id == "bad" {
                Err(Error::Backend("boom".to_string()))
            } else {
                Ok(response("remote answer"))
            }
        })
        .await;
    assert_eq!(result.unwrap().content, "remote answer");

    // Every worker down: None tells the caller to run locally.
    let result = registry
        .try_remote(&telemetry, "m1", |_| async {
            Err(Error::Backend("down".to_string()))
        })
        .await;
    assert!(result.is_none());

    let snapshot = telemetry.snapshot().await;
    assert_eq!(snapshot.errors_per_model.get("worker:bad"), Some(&2));
    assert_eq!(snapshot.errors_per_model.get("worker:good"), Some(&1));
}

#[tokio::test]
async fn session_state_is_isolated_through_runtime() {
    let runtime = Runtime::with_config(RuntimeConfig::default(), vec![mock_model()])
        .await
        .unwrap();
    let first = runtime.create_agent_session(vec!["calc".to_string()]).await;
    let second = runtime.create_agent_session(Vec::new()).await;
    assert_ne!(first, second);

    runtime
        .sessions()
        .spawn_agent(&first, "worker")
        .await
        .unwrap();
    runtime
        .sessions()
        .spawn_agent(&second, "worker")
        .await
        .unwrap();
    runtime
        .sessions()
        .push_agent_transcript(&first, "worker", "first-only".to_string())
        .await
        .unwrap();

    let task = runtime
        .sessions()
        .agent_task(&first, "worker")
        .await
        .unwrap();
    assert_eq!(task.transcript, vec!["first-only".to_string()]);
    let task = runtime
        .sessions()
        .agent_task(&second, "worker")
        .await
        .unwrap();
    assert!(task.transcript.is_empty());
    // Shared policy differs per session.
    assert!(runtime
        .sessions()
        .is_tool_allowed(&first, "calc")
        .await
        .unwrap());
    assert!(!runtime
        .sessions()
        .is_tool_allowed(&second, "calc")
        .await
        .unwrap());
}

#[tokio::test]
async fn run_agent_in_unknown_session_is_rejected() {
    let runtime = Runtime::with_config(RuntimeConfig::default(), Vec::new())
        .await
        .unwrap();
    let err = runtime
        .run_agent_in_session("no-such-session", "a", AgentRequest::new("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Validation(_)));
}

async fn mock_runtime(
    server: &MockServer,
    mutate: impl FnOnce(&mut RuntimeConfig),
) -> Arc<Runtime> {
    let mut config = RuntimeConfig {
        openai_base_url: server.base_url.clone(),
        llamacpp_base_url: server.base_url.clone(),
        request_timeout_secs: 10,
        max_retries: 1,
        retry_backoff_ms: 5,
        ..RuntimeConfig::default()
    };
    mutate(&mut config);
    Arc::new(
        Runtime::with_config(config, vec![mock_model()])
            .await
            .unwrap(),
    )
}

#[tokio::test]
async fn distributed_falls_back_to_local_when_worker_down() {
    let server = start_mock_server().await;
    // Worker points at a closed port; the local backend is the mock.
    let runtime = mock_runtime(&server, |config| {
        config.workers = vec![worker("w0", "http://127.0.0.1:9", &["mock-gpt"])];
    })
    .await;

    let response = runtime.infer(test_request("hello")).await.unwrap();
    assert!(response.content.contains("mock-echo: hello"));

    let status = runtime.get_status().await;
    assert_eq!(status["distributed"]["enabled"], true);
    let telemetry = status["telemetry"].clone();
    let worker_errors = telemetry["errors_per_model"]["worker:w0"]
        .as_u64()
        .unwrap_or(0);
    assert!(worker_errors >= 1, "telemetry: {telemetry}");
    server.shutdown().await;
}

#[tokio::test]
async fn worker_serves_requests_when_healthy() {
    let server = start_mock_server().await;
    // Local backends are unreachable; only the worker (the mock) can answer.
    let runtime = mock_runtime(&server, |config| {
        config.openai_base_url = "http://127.0.0.1:9".to_string();
        config.workers = vec![worker("w0", &server.base_url, &["mock-gpt"])];
    })
    .await;

    let before = server.state.chat_calls.load(Ordering::SeqCst);
    let response = runtime.infer(test_request("via worker")).await.unwrap();
    assert!(response.content.contains("mock-echo: via worker"));
    assert!(server.state.chat_calls.load(Ordering::SeqCst) > before);

    let status = runtime.get_status().await;
    assert!(status["telemetry"]["errors_per_model"]
        .get("worker:w0")
        .is_none());
    server.shutdown().await;
}

#[tokio::test]
async fn worker_serves_streams_when_healthy() {
    use futures_util::StreamExt;
    let server = start_mock_server().await;
    // Local backends are unreachable; only the worker (the mock) can stream.
    let runtime = mock_runtime(&server, |config| {
        config.openai_base_url = "http://127.0.0.1:9".to_string();
        config.workers = vec![worker("w0", &server.base_url, &["mock-gpt"])];
    })
    .await;

    let mut request = test_request("stream via worker");
    request.stream = Some(true);
    let mut stream = runtime.infer_stream(request).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(&chunk.unwrap());
    }
    assert!(text.contains("Hello"), "{text}");
    assert!(text.contains("world"), "{text}");

    let status = runtime.get_status().await;
    assert!(status["telemetry"]["errors_per_model"]
        .get("worker:w0")
        .is_none());
    server.shutdown().await;
}

#[tokio::test]
async fn session_run_mirrors_state_and_enforces_shared_policy() {
    let server = start_mock_server().await;
    let runtime = mock_runtime(&server, |_| {}).await;

    // Deny-all session strips the requested tool: without tools the mock
    // answers directly, so the run succeeds.
    let denied = runtime.create_agent_session(Vec::new()).await;
    let request = AgentRequest::new("api hello").with_allowed_tools(vec!["echo".to_string()]);
    let response = runtime
        .run_agent_in_session(&denied, "a", request)
        .await
        .unwrap();
    assert!(response.is_success());
    let task = runtime.sessions().agent_task(&denied, "a").await.unwrap();
    assert_eq!(task.state, AgentState::Done);
    assert_eq!(task.steps_used, 1);
    assert!(task
        .transcript
        .iter()
        .any(|line| line.starts_with("user: ")));
    assert!(task
        .transcript
        .iter()
        .any(|line| line.starts_with("assistant: ")));

    // Session policy allows echo through: the mock then demands an echo
    // tool call with invalid args, so the run fails at tool validation —
    // proving the tool was offered to the model.
    let allowed = runtime.create_agent_session(vec!["echo".to_string()]).await;
    let request = AgentRequest::new("api hello").with_allowed_tools(vec!["echo".to_string()]);
    let response = runtime
        .run_agent_in_session(&allowed, "a", request)
        .await
        .unwrap();
    assert_eq!(response.state, AgentState::Failed);
    let task = runtime.sessions().agent_task(&allowed, "a").await.unwrap();
    assert_eq!(task.state, AgentState::Failed);

    server.shutdown().await;
}
