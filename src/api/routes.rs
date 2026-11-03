use crate::api::{dashboard, handlers};
use crate::core::runtime::Runtime;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn create_router(state: Arc<Runtime>) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/status", get(handlers::status))
        .route("/v1/infer", post(handlers::infer))
        .route("/v1/infer/stream", post(handlers::infer_stream))
        .route("/v1/agent/run", post(handlers::run_agent))
        .route("/v1/agent/tools", get(handlers::list_agent_tools))
        .route("/dashboard", get(dashboard::dashboard))
        .fallback(handlers::fallback_404)
        .method_not_allowed_fallback(handlers::fallback_405)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
