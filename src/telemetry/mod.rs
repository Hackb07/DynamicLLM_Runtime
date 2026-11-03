use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub request_id: String,
    pub model: Option<String>,
    pub candidates: Vec<String>,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub success: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TelemetrySnapshot {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub in_flight_requests: u64,
    pub avg_latency_ms: Option<f64>,
    pub requests_per_model: HashMap<String, u64>,
    pub errors_per_model: HashMap<String, u64>,
    pub model_loads: HashMap<String, u64>,
    /// Measured mean latency per model (ms), from finished requests.
    #[serde(default)]
    pub avg_latency_ms_per_model: HashMap<String, f64>,
    /// Measured success rate per model: ok / (ok + err).
    #[serde(default)]
    pub success_rate_per_model: HashMap<String, f64>,
    /// Prefetch predictor outcomes (see `record_prefetch_outcome`).
    #[serde(default)]
    pub prefetch_hits: u64,
    #[serde(default)]
    pub prefetch_total: u64,
    #[serde(default)]
    pub prefetch_hit_rate: Option<f64>,
}

/// Per-model quality signals for adaptive routing. All values are
/// measured from finished requests; `None` means "no samples yet" and
/// callers should use a neutral prior rather than zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelStats {
    pub avg_latency_ms: Option<f64>,
    pub success_rate: Option<f64>,
    pub samples: u64,
}

pub struct Telemetry {
    in_flight: Arc<RwLock<HashMap<String, RequestMetrics>>>,
    total_requests: Arc<RwLock<u64>>,
    completed_requests: Arc<RwLock<u64>>,
    failed_requests: Arc<RwLock<u64>>,
    total_latency: Arc<RwLock<Duration>>,
    requests_per_model: Arc<RwLock<HashMap<String, u64>>>,
    errors_per_model: Arc<RwLock<HashMap<String, u64>>>,
    model_loads: Arc<RwLock<HashMap<String, u64>>>,
    latency_sum_ms_per_model: Arc<RwLock<HashMap<String, f64>>>,
    latency_counts_per_model: Arc<RwLock<HashMap<String, u64>>>,
    prefetch_hits: Arc<RwLock<u64>>,
    prefetch_total: Arc<RwLock<u64>>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            in_flight: Arc::new(RwLock::new(HashMap::new())),
            total_requests: Arc::new(RwLock::new(0)),
            completed_requests: Arc::new(RwLock::new(0)),
            failed_requests: Arc::new(RwLock::new(0)),
            total_latency: Arc::new(RwLock::new(Duration::ZERO)),
            requests_per_model: Arc::new(RwLock::new(HashMap::new())),
            errors_per_model: Arc::new(RwLock::new(HashMap::new())),
            model_loads: Arc::new(RwLock::new(HashMap::new())),
            latency_sum_ms_per_model: Arc::new(RwLock::new(HashMap::new())),
            latency_counts_per_model: Arc::new(RwLock::new(HashMap::new())),
            prefetch_hits: Arc::new(RwLock::new(0)),
            prefetch_total: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn record_request_start(&self, request_id: &str) {
        *self.total_requests.write().await += 1;
        self.in_flight.write().await.insert(
            request_id.to_string(),
            RequestMetrics {
                request_id: request_id.to_string(),
                model: None,
                candidates: Vec::new(),
                start_time: Instant::now(),
                end_time: None,
                success: None,
                error: None,
            },
        );
    }

    pub async fn record_routing(&self, request_id: &str, candidates: &[String]) {
        if let Some(metrics) = self.in_flight.write().await.get_mut(request_id) {
            metrics.candidates = candidates.to_vec();
            tracing::debug!("Request {request_id} candidates: {candidates:?}");
        }
    }

    pub async fn record_request_end(&self, request_id: &str, model: &str, success: bool) {
        let mut in_flight = self.in_flight.write().await;
        if let Some(metrics) = in_flight.remove(request_id) {
            let latency = metrics.start_time.elapsed();
            drop(in_flight);
            self.observe_finished_request(model, latency, success).await;
            if success {
                tracing::info!("Request {request_id} served by {model} in {latency:?}");
            } else {
                tracing::warn!("Request {request_id} failed on {model} after {latency:?}");
            }
        }
    }

    /// Shared accounting for one finished request (measured latency +
    /// per-model success counts). The `none` pseudo-model used for
    /// routing failures is counted globally but excluded from per-model
    /// quality signals so it never poisons routing scores.
    async fn observe_finished_request(&self, model: &str, latency: Duration, success: bool) {
        *self.total_latency.write().await += latency;
        if success {
            *self.completed_requests.write().await += 1;
        } else {
            *self.failed_requests.write().await += 1;
        }
        if model == "none" {
            return;
        }
        if success {
            *self
                .requests_per_model
                .write()
                .await
                .entry(model.to_string())
                .or_insert(0) += 1;
        } else {
            *self
                .errors_per_model
                .write()
                .await
                .entry(model.to_string())
                .or_insert(0) += 1;
        }
        *self
            .latency_sum_ms_per_model
            .write()
            .await
            .entry(model.to_string())
            .or_insert(0.0) += latency.as_secs_f64() * 1000.0;
        *self
            .latency_counts_per_model
            .write()
            .await
            .entry(model.to_string())
            .or_insert(0) += 1;
    }

    /// Record a synthetic observation (deterministic latency + outcome).
    /// Used by tests and benchmarks to seed routing scores without
    /// depending on wall-clock timing.
    pub async fn record_observation(&self, model: &str, latency_ms: f64, success: bool) {
        self.observe_finished_request(model, Duration::from_secs_f64(latency_ms / 1000.0), success)
            .await;
    }

    /// Measured quality signals for one model. Returns `None` fields
    /// when the model has no finished requests yet.
    pub async fn model_stats(&self, model: &str) -> ModelStats {
        let ok = self
            .requests_per_model
            .read()
            .await
            .get(model)
            .copied()
            .unwrap_or(0);
        let err = self
            .errors_per_model
            .read()
            .await
            .get(model)
            .copied()
            .unwrap_or(0);
        let total = ok + err;
        let success_rate = if total > 0 {
            Some(ok as f64 / total as f64)
        } else {
            None
        };
        let avg_latency_ms = {
            let sums = self.latency_sum_ms_per_model.read().await;
            let counts = self.latency_counts_per_model.read().await;
            match (sums.get(model), counts.get(model).copied().unwrap_or(0)) {
                (Some(sum), count) if count > 0 => Some(sum / count as f64),
                _ => None,
            }
        };
        ModelStats {
            avg_latency_ms,
            success_rate,
            samples: total,
        }
    }

    /// Record one prefetch-predictor outcome: whether the predicted
    /// next model matched the model actually demanded next. Hit rate
    /// is `hits / total` once any predictions were scored.
    pub async fn record_prefetch_outcome(&self, hit: bool) {
        *self.prefetch_total.write().await += 1;
        if hit {
            *self.prefetch_hits.write().await += 1;
        }
    }

    pub async fn prefetch_hit_rate(&self) -> Option<f64> {
        let total = *self.prefetch_total.read().await;
        if total == 0 {
            None
        } else {
            Some(*self.prefetch_hits.read().await as f64 / total as f64)
        }
    }

    /// Note a remote-worker failure (opt-in remote execution): counted
    /// under `worker:<id>` so fallbacks to local execution stay visible
    /// in the snapshot without failing the request.
    pub async fn record_worker_failure(&self, worker_id: &str) {
        let label = format!("worker:{worker_id}");
        *self
            .errors_per_model
            .write()
            .await
            .entry(label)
            .or_insert(0) += 1;
    }

    pub async fn record_error(&self, request_id: &str, model: &str, error: &str) {
        if let Some(metrics) = self.in_flight.write().await.get_mut(request_id) {
            metrics.model = Some(model.to_string());
            metrics.error = Some(error.to_string());
        }
    }

    pub async fn record_model_load_start(&self, model_id: &str) {
        tracing::info!("Model load started: {model_id}");
    }

    pub async fn record_model_load_end(&self, model_id: &str, success: bool) {
        if success {
            *self
                .model_loads
                .write()
                .await
                .entry(model_id.to_string())
                .or_insert(0) += 1;
            tracing::info!("Model load completed: {model_id}");
        } else {
            tracing::warn!("Model load failed: {model_id}");
        }
    }

    pub async fn snapshot(&self) -> TelemetrySnapshot {
        let total = *self.total_requests.read().await;
        let completed = *self.completed_requests.read().await;
        let failed = *self.failed_requests.read().await;
        let latency = *self.total_latency.read().await;
        let finished = completed + failed;
        let requests = self.requests_per_model.read().await.clone();
        let errors = self.errors_per_model.read().await.clone();
        let sums = self.latency_sum_ms_per_model.read().await.clone();
        let counts = self.latency_counts_per_model.read().await.clone();
        let mut avg_latency_ms_per_model = HashMap::new();
        for (model, sum) in &sums {
            if let Some(count) = counts.get(model).copied().filter(|c| *c > 0) {
                avg_latency_ms_per_model.insert(model.clone(), sum / count as f64);
            }
        }
        let mut success_rate_per_model = HashMap::new();
        for model in requests.keys().chain(errors.keys()) {
            let ok = requests.get(model).copied().unwrap_or(0);
            let err = errors.get(model).copied().unwrap_or(0);
            if ok + err > 0 {
                success_rate_per_model.insert(model.clone(), ok as f64 / (ok + err) as f64);
            }
        }
        let prefetch_hits = *self.prefetch_hits.read().await;
        let prefetch_total = *self.prefetch_total.read().await;
        TelemetrySnapshot {
            total_requests: total,
            completed_requests: completed,
            failed_requests: failed,
            in_flight_requests: total.saturating_sub(finished),
            avg_latency_ms: if finished > 0 {
                Some(latency.as_secs_f64() * 1000.0 / finished as f64)
            } else {
                None
            },
            requests_per_model: requests,
            errors_per_model: errors,
            model_loads: self.model_loads.read().await.clone(),
            avg_latency_ms_per_model,
            success_rate_per_model,
            prefetch_hits,
            prefetch_total,
            prefetch_hit_rate: if prefetch_total > 0 {
                Some(prefetch_hits as f64 / prefetch_total as f64)
            } else {
                None
            },
        }
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn snapshot_counts_requests_and_latency() {
        let telemetry = Telemetry::new();
        telemetry.record_request_start("a").await;
        telemetry.record_request_end("a", "m1", true).await;
        telemetry.record_request_start("b").await;
        telemetry.record_request_end("b", "m1", false).await;

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.total_requests, 2);
        assert_eq!(snapshot.completed_requests, 1);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.in_flight_requests, 0);
        assert!(snapshot.avg_latency_ms.is_some());
        assert_eq!(snapshot.requests_per_model.get("m1"), Some(&1));
        assert_eq!(snapshot.errors_per_model.get("m1"), Some(&1));
    }

    #[tokio::test]
    async fn model_stats_and_prefetch_hit_rate() {
        let telemetry = Telemetry::new();
        assert_eq!(
            telemetry.model_stats("ghost").await,
            ModelStats {
                avg_latency_ms: None,
                success_rate: None,
                samples: 0,
            }
        );
        assert_eq!(telemetry.prefetch_hit_rate().await, None);
        telemetry.record_observation("m1", 100.0, true).await;
        telemetry.record_observation("m1", 300.0, false).await;
        let stats = telemetry.model_stats("m1").await;
        assert_eq!(stats.samples, 2);
        assert!((stats.avg_latency_ms.unwrap() - 200.0).abs() < 1e-6);
        assert!((stats.success_rate.unwrap() - 0.5).abs() < 1e-9);
        telemetry.record_prefetch_outcome(true).await;
        telemetry.record_prefetch_outcome(false).await;
        assert!((telemetry.prefetch_hit_rate().await.unwrap() - 0.5).abs() < 1e-9);
        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.prefetch_hits, 1);
        assert_eq!(snapshot.prefetch_total, 2);
        assert!(snapshot.avg_latency_ms_per_model.contains_key("m1"));
        assert!(snapshot.success_rate_per_model.contains_key("m1"));
    }
}
