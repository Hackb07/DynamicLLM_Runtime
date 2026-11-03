//! Shared mock OpenAI-compatible backend for integration tests.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[derive(Default)]
pub struct MockState {
    pub chat_calls: AtomicU32,
    /// Next N chat calls fail with HTTP 500 before succeeding.
    pub fail_next: AtomicU32,
    pub last_model: std::sync::Mutex<Option<String>>,
    pub saw_auth: AtomicU32,
}

pub struct MockServer {
    pub base_url: String,
    // Not every test binary reads counters; silence per-binary dead-code lint.
    #[allow(dead_code)]
    pub state: Arc<MockState>,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockServer {
    pub async fn shutdown(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

pub async fn start_mock_server() -> MockServer {
    let state = Arc::new(MockState::default());
    let app = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/models", get(models))
        .route("/health", get(health))
        .with_state(state.clone());

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

    MockServer {
        base_url,
        state,
        shutdown: Some(shutdown_tx),
        handle: Some(handle),
    }
}

async fn models() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "data": [{ "id": "mock-gpt" }] }))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn chat_completions(
    State(state): State<Arc<MockState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    state.chat_calls.fetch_add(1, Ordering::SeqCst);
    if headers.contains_key("authorization") {
        state.saw_auth.fetch_add(1, Ordering::SeqCst);
    }
    let model = body
        .get("model")
        .and_then(|model| model.as_str())
        .unwrap_or("mock-gpt")
        .to_string();
    *state.last_model.lock().unwrap() = Some(model.clone());

    if state.fail_next.load(Ordering::SeqCst) > 0 {
        state.fail_next.fetch_sub(1, Ordering::SeqCst);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "boom" })),
        )
            .into_response();
    }
    if model == "bad-request" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "invalid model" })),
        )
            .into_response();
    }
    if model == "slow" {
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    let prompt = body
        .get("messages")
        .and_then(|messages| messages.get(0))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string();

    if body.get("stream").and_then(|stream| stream.as_bool()) == Some(true) {
        let chunks = vec!["Hello", " world", "!"];
        let stream = futures_util::stream::iter(chunks.into_iter().map(|text| {
            let payload = serde_json::json!({
                "choices": [{ "delta": { "content": text } }]
            });
            Ok::<Event, Infallible>(Event::default().data(payload.to_string()))
        }));
        // Note: axum's Sse formats `data: ...` lines; append [DONE] manually.
        let done = futures_util::stream::iter(vec![Ok::<Event, Infallible>(
            Event::default().data("[DONE]"),
        )]);
        let combined = futures_util::StreamExt::chain(stream, done);
        return Sse::new(combined).into_response();
    }

    let has_tools = body
        .get("tools")
        .and_then(|tools| tools.as_array())
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);
    let first_tool = body
        .get("tools")
        .and_then(|tools| tools.get(0))
        .and_then(|tool| tool.get("function"))
        .and_then(|function| function.get("name"))
        .and_then(|name| name.as_str())
        .unwrap_or("tool")
        .to_string();

    let message = if has_tools {
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": first_tool,
                    "arguments": "{\"x\":1}"
                }
            }]
        })
    } else {
        serde_json::json!({
            "role": "assistant",
            "content": format!("mock-echo: {prompt}")
        })
    };
    let finish_reason = if has_tools { "tool_calls" } else { "stop" };
    Json(serde_json::json!({
        "id": "chatcmpl-mock",
        "model": model,
        "choices": [{ "message": message, "finish_reason": finish_reason }],
        "usage": {
            "prompt_tokens": prompt.len() as u32 / 4,
            "completion_tokens": 8,
            "total_tokens": prompt.len() as u32 / 4 + 8
        }
    }))
    .into_response()
}

/// Prompt-only request builder for tests.
pub fn test_request(prompt: &str) -> dynamic_llm_runtime::InferenceRequest {
    dynamic_llm_runtime::InferenceRequest {
        request_id: None,
        prompt: prompt.to_string(),
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
