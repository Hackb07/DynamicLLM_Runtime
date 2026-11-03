//! Shared HTTP client for OpenAI-compatible chat-completions APIs.
//!
//! Both the OpenAI adapter and the llama.cpp-server adapter speak the same
//! wire format (`POST {base}{chat_path}` with `messages`), differing only in
//! base URL, auth, and health endpoint. Retries, timeouts, and SSE parsing
//! live here so every adapter gets them.

use crate::adapters::base::ChunkStream;
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse};
use futures_util::stream;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct FunctionDef {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct ToolDef {
    #[serde(rename = "type")]
    kind: String,
    function: FunctionDef,
}

#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDef>>,
    stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<UsageOut>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub message: Option<MessageOut>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageOut {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallOut>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallOut {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub function: FunctionCallOut,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FunctionCallOut {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageOut {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// Result of parsing one SSE `data:` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseParse {
    /// A text delta to forward to the caller.
    Content(String),
    /// The `[DONE]` terminator.
    Done,
    /// Anything else (comments, `event:` lines, role-only or usage chunks).
    Ignore,
}

/// Parse one raw SSE line (without the trailing newline).
pub fn parse_sse_line(line: &str) -> SseParse {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return SseParse::Ignore;
    }
    let Some(payload) = line.strip_prefix("data:") else {
        return SseParse::Ignore;
    };
    let payload = payload.trim();
    if payload == "[DONE]" {
        return SseParse::Done;
    }
    let value: serde_json::Value = match serde_json::from_str(payload) {
        Ok(value) => value,
        Err(_) => return SseParse::Ignore,
    };
    if value.get("error").is_some() {
        // Surface backend-reported stream errors instead of dropping them.
        return SseParse::Content(format!("\n[backend error: {value}]"));
    }
    let text = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default();
    if text.is_empty() {
        SseParse::Ignore
    } else {
        SseParse::Content(text.to_string())
    }

    // Note: kept as manual `Value` traversal (rather than `StreamChunk`)
    // so unknown extra fields from compatible servers never break parsing.
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Retryable HTTP statuses: rate limiting plus server-side failures.
fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn map_transport_error(what: &str, err: reqwest::Error) -> Error {
    if err.is_timeout() {
        Error::Timeout
    } else if err.is_connect() {
        Error::Backend(format!("{what}: connection failed: {err}"))
    } else if err.is_body() || err.is_decode() {
        Error::Backend(format!("{what}: invalid response body: {err}"))
    } else {
        Error::Backend(format!("{what}: {err}"))
    }
}

/// An HTTP client for one OpenAI-compatible backend.
#[derive(Debug, Clone)]
pub struct OpenAiCompatClient {
    http: Client,
    base_url: String,
    chat_path: String,
    health_path: String,
    api_key: Option<String>,
    backend_label: String,
    max_retries: u32,
    backoff_base_ms: u64,
}

impl OpenAiCompatClient {
    // Eight cohesive parameters beat a builder for an internal constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend_label: impl Into<String>,
        base_url: impl Into<String>,
        chat_path: impl Into<String>,
        health_path: impl Into<String>,
        api_key: Option<String>,
        timeout: Duration,
        max_retries: u32,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Backend(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            chat_path: chat_path.into(),
            health_path: health_path.into(),
            api_key,
            backend_label: backend_label.into(),
            max_retries,
            backoff_base_ms,
        })
    }

    fn chat_url(&self) -> String {
        format!("{}{}", self.base_url, self.chat_path)
    }

    fn health_url(&self) -> String {
        format!("{}{}", self.base_url, self.health_path)
    }

    fn request_body(model: &str, request: &InferenceRequest, stream: bool) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: request.prompt.clone(),
            }],
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop.clone(),
            tools: request.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .map(|tool| ToolDef {
                        kind: "function".to_string(),
                        function: FunctionDef {
                            name: tool.name.clone(),
                            description: Some(tool.description.clone()),
                            parameters: tool.parameters.clone(),
                        },
                    })
                    .collect()
            }),
            stream,
        }
    }

    fn authorized(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.header("Authorization", format!("Bearer {key}")),
            None => builder,
        }
    }

    /// POST one chat-completions request with retries on transient failures.
    pub async fn chat(&self, model: &str, request: &InferenceRequest) -> Result<ChatResponse> {
        let body = Self::request_body(model, request, false);
        let mut attempt = 0u32;
        loop {
            let send = self
                .authorized(self.http.post(self.chat_url()).json(&body))
                .send()
                .await;
            match send {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return response.json::<ChatResponse>().await.map_err(|e| {
                            Error::Backend(format!(
                                "{}: invalid JSON response: {e}",
                                self.backend_label
                            ))
                        });
                    }
                    let text = response.text().await.unwrap_or_default();
                    let failed = Error::Backend(format!(
                        "{} returned HTTP {status}: {text}",
                        self.backend_label
                    ));
                    if attempt < self.max_retries && is_retryable_status(status) {
                        attempt += 1;
                        self.backoff(attempt).await;
                        continue;
                    }
                    return Err(failed);
                }
                Err(e) if e.is_timeout() => return Err(Error::Timeout),
                Err(e) => {
                    // Connect errors and other transport failures are retried;
                    // body/decode errors surface immediately via fallthrough.
                    let retryable = e.is_connect() || e.is_request();
                    if attempt < self.max_retries && retryable {
                        attempt += 1;
                        self.backoff(attempt).await;
                        continue;
                    }
                    return Err(map_transport_error(&self.backend_label, e));
                }
            }
        }
    }

    /// POST a streaming chat-completions request. The response body is read
    /// incrementally with `Response::chunk` so TTFT stays low; SSE `data:`
    /// lines are parsed into text deltas. Single attempt (documented):
    /// retrying a partially-consumed stream would duplicate tokens.
    pub async fn chat_stream(
        &self,
        model: &str,
        request: &InferenceRequest,
    ) -> Result<ChunkStream> {
        let body = Self::request_body(model, request, true);
        let response = self
            .authorized(self.http.post(self.chat_url()).json(&body))
            .send()
            .await
            .map_err(|e| map_transport_error(&self.backend_label, e))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::Backend(format!(
                "{} returned HTTP {status}: {text}",
                self.backend_label
            )));
        }
        let label = self.backend_label.clone();
        let state = SseStreamState {
            response,
            buffer: String::new(),
            finished: false,
        };
        let s = stream::unfold(state, move |mut state| {
            let label = label.clone();
            async move {
                if state.finished {
                    return None;
                }
                loop {
                    if let Some(line) = take_line(&mut state.buffer) {
                        match parse_sse_line(&line) {
                            SseParse::Content(text) => {
                                return Some((Ok(text), state));
                            }
                            SseParse::Done => return None,
                            SseParse::Ignore => continue,
                        }
                    }
                    match state.response.chunk().await {
                        Ok(Some(bytes)) => {
                            state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => {
                            // Flush any unterminated trailing data, then end.
                            let rest = std::mem::take(&mut state.buffer);
                            return match parse_sse_line(rest.trim()) {
                                SseParse::Content(text) => {
                                    state.finished = true;
                                    Some((Ok(text), state))
                                }
                                _ => None,
                            };
                        }
                        Err(e) => {
                            state.finished = true;
                            return Some((Err(map_transport_error(&label, e)), state));
                        }
                    }
                }
            }
        });
        Ok(Box::pin(s))
    }

    /// Best-effort reachability probe. Never fails: returns `false` when the
    /// backend is unreachable or reports an error status.
    pub async fn health(&self) -> bool {
        let send = self.authorized(self.http.get(self.health_url())).send();
        match tokio::time::timeout(Duration::from_secs(5), send).await {
            Ok(Ok(response)) => response.status().is_success(),
            _ => false,
        }
    }

    async fn backoff(&self, attempt: u32) {
        let delay_ms = self.backoff_base_ms.saturating_mul(1u64 << attempt.min(6));
        tracing::debug!(
            "{}: retry attempt {attempt} after {delay_ms}ms",
            self.backend_label
        );
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }
}

struct SseStreamState {
    response: reqwest::Response,
    buffer: String,
    finished: bool,
}

/// Map a chat-completions response onto the runtime's normalized response.
pub fn to_inference_response(
    request: &InferenceRequest,
    model: &str,
    response: ChatResponse,
) -> InferenceResponse {
    use crate::core::types::{ToolCall, Usage};

    let choice = response.choices.first();
    let message = choice.and_then(|choice| choice.message.as_ref());
    let content = message
        .and_then(|message| message.content.clone())
        .unwrap_or_default();
    let tool_calls = message.and_then(|message| {
        message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|call| ToolCall {
                    tool: call.function.name.clone(),
                    arguments: serde_json::from_str(&call.function.arguments)
                        .unwrap_or(serde_json::Value::String(call.function.arguments.clone())),
                })
                .collect()
        })
    });
    InferenceResponse {
        request_id: request
            .request_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        model: if response.model.is_empty() {
            model.to_string()
        } else {
            response.model
        },
        content,
        finish_reason: choice.and_then(|choice| choice.finish_reason.clone()),
        usage: response.usage.map(|usage| Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }),
        tool_calls,
        metadata: None,
    }
}

/// Split one `\n`-terminated line off the front of the buffer.
fn take_line(buffer: &mut String) -> Option<String> {
    let pos = buffer.find('\n')?;
    let line = buffer[..pos].to_string();
    buffer.drain(..=pos);
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(parse_sse_line(line), SseParse::Content("Hello".to_string()));
    }

    #[test]
    fn parses_done_terminator() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseParse::Done);
        assert_eq!(parse_sse_line("data:[DONE]\r"), SseParse::Done);
    }

    #[test]
    fn ignores_comments_events_and_empty_lines() {
        assert_eq!(parse_sse_line(""), SseParse::Ignore);
        assert_eq!(parse_sse_line(": ping"), SseParse::Ignore);
        assert_eq!(parse_sse_line("event: message"), SseParse::Ignore);
        assert_eq!(
            parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#),
            SseParse::Ignore
        );
    }

    #[test]
    fn ignores_malformed_payloads() {
        assert_eq!(parse_sse_line("data: not-json{"), SseParse::Ignore);
        assert_eq!(parse_sse_line("data: "), SseParse::Ignore);
    }

    #[test]
    fn surfaces_backend_stream_errors() {
        let line = r#"data: {"error":{"message":"overloaded"}}"#;
        match parse_sse_line(line) {
            SseParse::Content(text) => assert!(text.contains("overloaded")),
            other => panic!("expected Content, got {other:?}"),
        }
    }

    #[test]
    fn take_line_splits_and_drains() {
        let mut buffer = "data: a\ndata: b".to_string();
        assert_eq!(take_line(&mut buffer).as_deref(), Some("data: a"));
        assert_eq!(buffer, "data: b");
        assert_eq!(take_line(&mut buffer), None);
    }

    #[test]
    fn retryable_statuses() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(StatusCode::OK));
    }
}
