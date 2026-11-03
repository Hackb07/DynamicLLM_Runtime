mod common;

use common::{start_mock_server, test_request};
use dynamic_llm_runtime::{Adapter, Error, LlamaCppAdapter, OpenAIAdapter};
use futures_util::StreamExt;
use std::sync::atomic::Ordering;
use std::time::Duration;

#[tokio::test]
async fn openai_infer_roundtrip() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        Some("test-key".to_string()),
        "mock-gpt",
        Duration::from_secs(10),
        0,
        10,
    )
    .unwrap();

    let mut request = test_request("hello world");
    request.model = Some("mock-gpt".to_string());
    let response = adapter.infer(&request).await.unwrap();

    assert!(response.content.contains("mock-echo: hello world"));
    assert_eq!(response.model, "mock-gpt");
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    let usage = response.usage.expect("usage present");
    assert!(usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens);
    assert!(!response.request_id.is_empty());
    assert_eq!(server.state.saw_auth.load(Ordering::SeqCst), 1);

    server.shutdown().await;
}

#[tokio::test]
async fn openai_infer_maps_tool_calls() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_secs(10),
        0,
        10,
    )
    .unwrap();

    let mut request = test_request("use the tool");
    request.model = Some("mock-gpt".to_string());
    request.tools = Some(vec![dynamic_llm_runtime::core::types::Tool {
        name: "get_weather".to_string(),
        description: "weather lookup".to_string(),
        parameters: serde_json::json!({ "type": "object" }),
    }]);
    let response = adapter.infer(&request).await.unwrap();

    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    let calls = response.tool_calls.expect("tool calls present");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool, "get_weather");
    assert_eq!(calls[0].arguments, serde_json::json!({ "x": 1 }));

    server.shutdown().await;
}

#[tokio::test]
async fn openai_stream_yields_ordered_deltas() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_secs(10),
        0,
        10,
    )
    .unwrap();

    let mut request = test_request("stream please");
    request.model = Some("mock-gpt".to_string());
    let mut stream = adapter.infer_stream(&request).await.unwrap();
    let mut joined = String::new();
    let mut count = 0;
    while let Some(chunk) = stream.next().await {
        joined.push_str(&chunk.unwrap());
        count += 1;
    }
    assert_eq!(joined, "Hello world!");
    assert_eq!(count, 3);

    server.shutdown().await;
}

#[tokio::test]
async fn openai_retries_transient_errors_then_succeeds() {
    let server = start_mock_server().await;
    server.state.fail_next.store(2, Ordering::SeqCst);
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_secs(10),
        3,
        5,
    )
    .unwrap();

    let mut request = test_request("retry me");
    request.model = Some("mock-gpt".to_string());
    let response = adapter.infer(&request).await.unwrap();
    assert!(response.content.contains("mock-echo"));
    assert_eq!(server.state.chat_calls.load(Ordering::SeqCst), 3);

    server.shutdown().await;
}

#[tokio::test]
async fn openai_does_not_retry_client_errors() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_secs(10),
        3,
        5,
    )
    .unwrap();

    let mut request = test_request("bad");
    request.model = Some("bad-request".to_string());
    let err = adapter.infer(&request).await.unwrap_err();
    assert!(matches!(err, Error::Backend(_)), "got {err:?}");
    assert_eq!(server.state.chat_calls.load(Ordering::SeqCst), 1);

    server.shutdown().await;
}

#[tokio::test]
async fn openai_timeout_surfaces_as_timeout() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_millis(100),
        0,
        5,
    )
    .unwrap();

    let mut request = test_request("slow");
    request.model = Some("slow".to_string());
    let err = adapter.infer(&request).await.unwrap_err();
    assert!(matches!(err, Error::Timeout), "got {err:?}");

    server.shutdown().await;
}

#[tokio::test]
async fn openai_health_reflects_reachability() {
    let server = start_mock_server().await;
    let adapter = OpenAIAdapter::from_config(
        &server.base_url,
        None,
        "mock-gpt",
        Duration::from_secs(5),
        0,
        5,
    )
    .unwrap();
    assert!(adapter.health_check().await.unwrap());

    let dead = OpenAIAdapter::from_config(
        "http://127.0.0.1:1",
        None,
        "mock-gpt",
        Duration::from_secs(5),
        0,
        5,
    )
    .unwrap();
    assert!(!dead.health_check().await.unwrap());

    server.shutdown().await;
}

#[tokio::test]
async fn llama_load_requires_a_live_server() {
    let server = start_mock_server().await;
    let adapter = LlamaCppAdapter::from_config(
        &server.base_url,
        "mock-llama",
        Duration::from_secs(10),
        0,
        5,
    )
    .unwrap();
    adapter.load().await.unwrap();

    let mut request = test_request("via v1 path");
    request.model = Some("mock-llama".to_string());
    let response = adapter.infer(&request).await.unwrap();
    assert!(response.content.contains("mock-echo"));

    let dead = LlamaCppAdapter::from_config(
        "http://127.0.0.1:1",
        "mock-llama",
        Duration::from_secs(5),
        0,
        5,
    )
    .unwrap();
    let err = dead.load().await.unwrap_err();
    assert!(matches!(err, Error::ResourceUnavailable(_)), "got {err:?}");

    server.shutdown().await;
}
