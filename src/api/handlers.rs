use crate::agents::{AgentRequest, AgentResponse, ToolDefinition};
use crate::core::error::Error;
use crate::core::runtime::Runtime;
use crate::core::types::{InferenceRequest, InferenceResponse};
use crate::models::registry::ModelInfo;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::Stream;
use futures_util::StreamExt;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;

/// Error type that renders backend failures as JSON with proper statuses.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// Map an axum JSON-extractor rejection onto the `{"error": msg}` shape
    /// so malformed bodies and wrong content types never bypass it.
    fn from_json_rejection(rejection: axum::extract::rejection::JsonRejection) -> Self {
        use axum::extract::rejection::JsonRejection;
        match rejection {
            JsonRejection::MissingJsonContentType(_) => Self::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "content-type must be application/json",
            ),
            other => Self::new(StatusCode::BAD_REQUEST, other.to_string()),
        }
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let status = match &error {
            Error::Validation(_) => StatusCode::BAD_REQUEST,
            Error::Forbidden(_) => StatusCode::FORBIDDEN,
            Error::ModelNotFound(_) => StatusCode::NOT_FOUND,
            Error::Timeout => StatusCode::GATEWAY_TIMEOUT,
            Error::ResourceUnavailable(_)
            | Error::ModelNotReady(_)
            | Error::Routing(_)
            | Error::Cancelled => StatusCode::SERVICE_UNAVAILABLE,
            Error::Backend(_) => StatusCode::BAD_GATEWAY,
            Error::Config(_) | Error::Other(_) | Error::Io(_) | Error::Serialization(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        Self::new(status, error.user_message())
    }
}

/// Fallback for unknown paths: same `{"error": msg}` shape as `ApiError`.
pub async fn fallback_404() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "unknown endpoint")
}

/// Fallback for known paths hit with the wrong method.
pub async fn fallback_405() -> ApiError {
    ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn list_models(State(runtime): State<Arc<Runtime>>) -> Json<Vec<ModelInfo>> {
    Json(runtime.list_models().await)
}

pub async fn status(State(runtime): State<Arc<Runtime>>) -> Json<serde_json::Value> {
    Json(runtime.get_status().await)
}

pub async fn infer(
    State(runtime): State<Arc<Runtime>>,
    body: Result<Json<InferenceRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<InferenceResponse>, ApiError> {
    let Json(request) = body.map_err(ApiError::from_json_rejection)?;
    runtime
        .infer(request)
        .await
        .map(Json)
        .map_err(ApiError::from)
}

/// Run one agentic task. The `AgentResponse` body (answer, transcript,
/// audit) is always returned for executed runs; terminal failures use the
/// matching status code so clients can branch on it.
pub async fn run_agent(
    State(runtime): State<Arc<Runtime>>,
    body: Result<Json<AgentRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<(StatusCode, Json<AgentResponse>), ApiError> {
    let Json(request) = body.map_err(ApiError::from_json_rejection)?;
    let response = runtime.run_agent(request).await.map_err(ApiError::from)?;
    let status = agent_status(&response);
    Ok((status, Json(response)))
}

pub async fn list_agent_tools(State(runtime): State<Arc<Runtime>>) -> Json<Vec<ToolDefinition>> {
    Json(runtime.list_agent_tools())
}

/// Map a terminal agent failure to the closest HTTP status. Mirrors the
/// `ApiError` mapping (validation 400, disallowed tool 403, timeout 504,
/// backend 502) while keeping the `AgentResponse` body intact.
fn agent_status(response: &AgentResponse) -> StatusCode {
    if response.is_success() {
        return StatusCode::OK;
    }
    let message = response.error.as_deref().unwrap_or_default();
    let lower = message.to_ascii_lowercase();
    if lower.contains("denied by allowlist") || lower.contains("unknown tool") {
        StatusCode::FORBIDDEN
    } else if lower.contains("timed out") || lower.contains("timeout") {
        StatusCode::GATEWAY_TIMEOUT
    } else if lower.contains("backend error") || lower.contains("empty final content") {
        StatusCode::BAD_GATEWAY
    } else if lower.contains("model not found") {
        StatusCode::NOT_FOUND
    } else if lower.contains("cancelled")
        || lower.contains("resource unavailable")
        || lower.contains("model not ready")
        || lower.contains("routing error")
    {
        StatusCode::SERVICE_UNAVAILABLE
    } else if lower.contains("configuration error")
        || lower.contains("io error")
        || lower.contains("serialization error")
    {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        // Step/token budgets, bad tool arguments, unreadable files, and
        // anything else the caller can fix by changing the request.
        StatusCode::BAD_REQUEST
    }
}

type SseStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// SSE contract: text deltas arrive as unnamed `data:` events, a terminal
/// `data: [DONE]` closes a successful stream, and mid-stream failures arrive
/// as one named `error` event (same text as the JSON `{"error"}` shape)
/// followed by `[DONE]` so clients always see a terminator.
pub async fn infer_stream(
    State(runtime): State<Arc<Runtime>>,
    body: Result<Json<InferenceRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Sse<SseStream>, ApiError> {
    let Json(request) = body.map_err(ApiError::from_json_rejection)?;
    let chunks = runtime
        .infer_stream(request)
        .await
        .map_err(ApiError::from)?;
    let events = chunks.map(|result| {
        Ok(match result {
            Ok(text) => Event::default().data(text),
            Err(e) => Event::default().event("error").data(e.user_message()),
        })
    });
    // OpenAI-style terminator so clients can detect a complete stream.
    let done = futures_util::stream::iter([Ok(Event::default().data("[DONE]"))]);
    let stream: SseStream = Box::pin(events.chain(done));
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
