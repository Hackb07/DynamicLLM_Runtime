use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub request_id: String,
    pub model: Option<String>,
    pub start_time: std::time::Instant,
    pub end_time: Option<std::time::Instant>,
}

pub struct Telemetry {
    metrics: Arc<RwLock<HashMap<String, RequestMetrics>>>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn record_request_start(&self, request_id: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.insert(
            request_id.to_string(),
            RequestMetrics {
                request_id: request_id.to_string(),
                model: None,
                start_time: std::time::Instant::now(),
                end_time: None,
            },
        );
        tracing::info!("Request started: {}", request_id);
    }

    pub async fn record_request_end(&self, request_id: &str, model: &str) {
        let mut metrics = self.metrics.write().await;
        if let Some(m) = metrics.get_mut(request_id) {
            m.end_time = Some(std::time::Instant::now());
            m.model = Some(model.to_string());
            let duration = m.end_time.unwrap().duration_since(m.start_time);
            tracing::info!("Request completed: {} in {:?}", request_id, duration);
        }
    }

    pub async fn record_model_load_start(&self, model_id: &str) {
        tracing::info!("Model load started: {}", model_id);
    }

    pub async fn record_model_load_end(&self, model_id: &str) {
        tracing::info!("Model load completed: {}", model_id);
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}
