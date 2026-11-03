mod common;

use common::{start_mock_server, test_request, MockServer};
use dynamic_llm_runtime::agents::{
    AgentExecutor, AgentRequest, AgentState, TaskController, ToolRegistry,
};
use dynamic_llm_runtime::config::RuntimeConfig;
use dynamic_llm_runtime::core::types::{InferenceResponse, ToolCall};
use dynamic_llm_runtime::models::capabilities::ModelCapabilities;
use dynamic_llm_runtime::models::registry::ModelInfo;
use dynamic_llm_runtime::Runtime;
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

async fn test_runtime(server: &MockServer) -> Arc<Runtime> {
    let config = RuntimeConfig {
        openai_base_url: server.base_url.clone(),
        llamacpp_base_url: server.base_url.clone(),
        request_timeout_secs: 10,
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

#[tokio::test]
async fn agent_direct_answer_through_real_runtime() {
    let server = start_mock_server().await;
    let runtime = test_runtime(&server).await;
    let executor = AgentExecutor::new();

    let response = executor
        .run(&runtime, AgentRequest::new("api hello"))
        .await
        .unwrap();

    assert!(response.is_success());
    assert!(response.answer.contains("mock-echo: api hello"));
    assert_eq!(response.steps_used, 1);

    server.shutdown().await;
}

#[tokio::test]
async fn agent_reads_workspace_file_through_tool_loop() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("facts.txt"), "the sky is blue").unwrap();
    let executor = AgentExecutor::with_workspace_root(dir.path().to_path_buf());

    let request =
        AgentRequest::new("read facts.txt").with_allowed_tools(vec!["read_file".to_string()]);
    let response = executor
        .run_with_infer(request, |req| async move {
            if req.prompt.contains("the sky is blue") {
                Ok(InferenceResponse {
                    request_id: "r".to_string(),
                    model: "mock".to_string(),
                    content: "The sky is blue.".to_string(),
                    finish_reason: Some("stop".to_string()),
                    usage: None,
                    tool_calls: None,
                    metadata: None,
                })
            } else {
                Ok(InferenceResponse {
                    request_id: "r".to_string(),
                    model: "mock".to_string(),
                    content: String::new(),
                    finish_reason: Some("tool_calls".to_string()),
                    usage: None,
                    tool_calls: Some(vec![ToolCall {
                        tool: "read_file".to_string(),
                        arguments: serde_json::json!({"path": "facts.txt"}),
                    }]),
                    metadata: None,
                })
            }
        })
        .await
        .unwrap();

    assert!(response.is_success());
    assert_eq!(response.answer, "The sky is blue.");
    assert_eq!(response.steps_used, 2);
}

#[tokio::test]
async fn agent_path_traversal_fails_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let executor = AgentExecutor::with_workspace_root(dir.path().to_path_buf());

    let request = AgentRequest::new("read it").with_allowed_tools(vec!["read_file".to_string()]);
    let response = executor
        .run_with_infer(request, |_| async {
            Ok(InferenceResponse {
                request_id: "r".to_string(),
                model: "mock".to_string(),
                content: String::new(),
                finish_reason: Some("tool_calls".to_string()),
                usage: None,
                tool_calls: Some(vec![ToolCall {
                    tool: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "../secret.txt"}),
                }]),
                metadata: None,
            })
        })
        .await
        .unwrap();

    assert_eq!(response.state, AgentState::Failed);
    assert!(response.error.unwrap().contains("escapes"));
}

#[test]
fn public_api_smoke() {
    // Existing public shapes keep working.
    let mut registry = ToolRegistry::new();
    assert!(registry.list().is_empty());
    registry.register_safe_subset();
    assert_eq!(registry.list().len(), 4);
    assert!(registry.get("calc").is_some());

    let task = TaskController::new(3);
    assert_eq!(task.max_steps(), 3);
    assert_eq!(task.state(), AgentState::Pending);

    let _executor = AgentExecutor::new();
    let _default: AgentExecutor = Default::default();

    assert_eq!(test_request("x").prompt, "x");
}
