//! Dashboard route test. Intentionally mock-free (no loopback bind):
//! the page is static, so a default runtime is enough.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use dynamic_llm_runtime::api::create_router;
use dynamic_llm_runtime::config::RuntimeConfig;
use dynamic_llm_runtime::Runtime;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn dashboard_serves_html() {
    let runtime = Arc::new(
        Runtime::with_config(RuntimeConfig::default(), Vec::new())
            .await
            .unwrap(),
    );
    let request = Request::builder()
        .uri("/dashboard")
        .body(Body::empty())
        .unwrap();
    let response = create_router(runtime).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/html"),
        "unexpected content type: {content_type}"
    );
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let html = String::from_utf8(body.to_vec()).unwrap();
    // The page polls the JSON endpoints it visualizes.
    assert!(html.contains("/v1/status"), "missing status poll");
    assert!(html.contains("/v1/models"), "missing models poll");
    assert!(html.contains("Telemetry"), "missing telemetry section");
    // Residency column reads the serialized `estimated_mb` key.
    assert!(html.contains("estimated_mb"), "missing residency key");
}
