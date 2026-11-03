//! Opt-in remote execution: worker registry + placement.
//!
//! Workers are opt-in via config (`workers`, empty by default). With no
//! workers configured every method below degrades to "no remote target"
//! and the runtime's local path runs exactly as before.

use crate::adapters::ChunkStream;
use crate::config::WorkerConfig;
use crate::core::error::Result;
use crate::core::types::InferenceResponse;
use crate::telemetry::Telemetry;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

fn default_worker_capacity() -> usize {
    4
}

/// One remote inference worker: address, the models it serves, and how
/// many concurrent requests it accepts. An empty `models` list marks a
/// generalist that accepts any model (ranked after explicit matches).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerInfo {
    pub id: String,
    pub addr: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_worker_capacity")]
    pub capacity: usize,
}

impl WorkerInfo {
    /// Whether this worker may serve `model`.
    pub fn serves(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|served| served == model)
    }

    /// Whether this worker explicitly lists `model` (placement affinity).
    pub fn has_affinity(&self, model: &str) -> bool {
        self.models.iter().any(|served| served == model)
    }

    pub fn effective_capacity(&self) -> usize {
        self.capacity.max(1)
    }
}

/// RAII slot: acquiring increments the worker's in-flight count for
/// least-loaded placement; dropping decrements it.
pub struct WorkerSlot {
    in_flight: Arc<Mutex<HashMap<String, usize>>>,
    worker_id: String,
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        if let Ok(mut counts) = self.in_flight.lock() {
            counts
                .entry(self.worker_id.clone())
                .and_modify(|in_flight| *in_flight = in_flight.saturating_sub(1));
        }
    }
}

/// Registry of remote workers with load-aware placement. A plain
/// `std::mutex` guards the counters: the critical sections are tiny and
/// never held across `.await`.
pub struct WorkerRegistry {
    workers: Vec<WorkerInfo>,
    in_flight: Arc<Mutex<HashMap<String, usize>>>,
}

impl WorkerRegistry {
    pub fn new(workers: Vec<WorkerInfo>) -> Self {
        Self {
            workers,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn from_config(workers: &[WorkerConfig]) -> Self {
        Self::new(
            workers
                .iter()
                .map(|worker| WorkerInfo {
                    id: worker.id.clone(),
                    addr: worker.addr.clone(),
                    models: worker.models.clone(),
                    capacity: worker.capacity,
                })
                .collect(),
        )
    }

    /// True when distributed inference is disabled (the default):
    /// callers must use the local path unchanged.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn workers(&self) -> Vec<WorkerInfo> {
        self.workers.clone()
    }

    pub fn get(&self, worker_id: &str) -> Option<WorkerInfo> {
        self.workers
            .iter()
            .find(|worker| worker.id == worker_id)
            .cloned()
    }

    fn lock_counts(&self) -> MutexGuard<'_, HashMap<String, usize>> {
        self.in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn in_flight_count(&self, worker_id: &str) -> usize {
        self.lock_counts().get(worker_id).copied().unwrap_or(0)
    }

    /// Hold one load slot on a worker for the duration of a remote call.
    pub fn acquire(&self, worker_id: &str) -> WorkerSlot {
        self.lock_counts()
            .entry(worker_id.to_string())
            .and_modify(|in_flight| *in_flight += 1)
            .or_insert(1);
        WorkerSlot {
            in_flight: self.in_flight.clone(),
            worker_id: worker_id.to_string(),
        }
    }

    /// Eligible workers for `model` in placement order: explicit model
    /// affinity first, then least-loaded (in-flight / capacity), then id.
    /// Workers that list models without `model` are ineligible; `None`
    /// matches every worker by load only.
    pub fn candidates(&self, model: Option<&str>) -> Vec<WorkerInfo> {
        let counts = self.lock_counts();
        let load_ratio = |worker: &WorkerInfo| {
            counts.get(&worker.id).copied().unwrap_or(0) as f64 / worker.effective_capacity() as f64
        };
        let mut eligible: Vec<&WorkerInfo> = self
            .workers
            .iter()
            .filter(|worker| model.is_none_or(|model| worker.serves(model)))
            .collect();
        eligible.sort_by(|a, b| {
            let affinity =
                |worker: &WorkerInfo| model.is_some_and(|model| worker.has_affinity(model));
            affinity(b)
                .cmp(&affinity(a))
                .then_with(|| {
                    load_ratio(a)
                        .partial_cmp(&load_ratio(b))
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| a.id.cmp(&b.id))
        });
        eligible.into_iter().cloned().collect()
    }

    /// Try remote workers in placement order via `call`. Returns the first
    /// remote success, or `None` when no workers are configured or every
    /// worker failed — the caller then falls back to local execution.
    /// Each worker failure records a `worker:<id>` telemetry note.
    pub async fn try_remote<F, Fut>(
        &self,
        telemetry: &Telemetry,
        model: &str,
        mut call: F,
    ) -> Option<InferenceResponse>
    where
        F: FnMut(WorkerInfo) -> Fut,
        Fut: std::future::Future<Output = Result<InferenceResponse>>,
    {
        for worker in self.candidates(Some(model)) {
            let _slot = self.acquire(&worker.id);
            match call(worker.clone()).await {
                Ok(response) => return Some(response),
                Err(error) => {
                    tracing::warn!(
                        "worker {} failed for model {model}: {error}; trying next target",
                        worker.id
                    );
                    telemetry.record_worker_failure(&worker.id).await;
                }
            }
        }
        None
    }

    /// Streaming twin of [`Self::try_remote`]: tries remote workers in
    /// placement order and returns the first usable chunk stream, or `None`
    /// when no workers are configured or every worker failed to start a
    /// stream — the caller then falls back to local execution. Only
    /// stream *startup* fails over; once chunks flow, the stream stays put.
    pub async fn try_remote_stream<F, Fut>(
        &self,
        telemetry: &Telemetry,
        model: &str,
        mut call: F,
    ) -> Option<ChunkStream>
    where
        F: FnMut(WorkerInfo) -> Fut,
        Fut: std::future::Future<Output = Result<ChunkStream>>,
    {
        for worker in self.candidates(Some(model)) {
            // The slot only covers stream startup (the returned stream
            // outlives this scope); steady-state load is tracked by the
            // non-streaming path and the worker's own admission control.
            let _slot = self.acquire(&worker.id);
            match call(worker.clone()).await {
                Ok(stream) => return Some(stream),
                Err(error) => {
                    tracing::warn!(
                        "worker {} stream failed for model {model}: {error}; trying next target",
                        worker.id
                    );
                    telemetry.record_worker_failure(&worker.id).await;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, models: &[&str], capacity: usize) -> WorkerInfo {
        WorkerInfo {
            id: id.to_string(),
            addr: format!("http://{id}:8080"),
            models: models.iter().map(|model| model.to_string()).collect(),
            capacity,
        }
    }

    fn response(content: &str) -> InferenceResponse {
        InferenceResponse {
            request_id: "r".to_string(),
            model: "m1".to_string(),
            content: content.to_string(),
            finish_reason: Some("stop".to_string()),
            usage: None,
            tool_calls: None,
            metadata: None,
        }
    }

    fn ids(workers: &[WorkerInfo]) -> Vec<&str> {
        workers.iter().map(|worker| worker.id.as_str()).collect()
    }

    #[test]
    fn empty_registry_has_no_candidates() {
        let registry = WorkerRegistry::new(Vec::new());
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.candidates(Some("m1")).is_empty());
        assert!(registry.candidates(None).is_empty());
    }

    #[test]
    fn placement_prefers_affinity_then_least_loaded() {
        let registry = WorkerRegistry::new(vec![
            worker("generalist", &[], 10),
            worker("other", &["m2"], 10),
            worker("busy-match", &["m1"], 10),
            worker("idle-match", &["m1"], 10),
        ]);
        // Load the first affinity match so ordering within the tier matters.
        let _slots: Vec<WorkerSlot> = (0..9).map(|_| registry.acquire("busy-match")).collect();

        let ordered = registry.candidates(Some("m1"));
        // Explicit affinity first (least-loaded within the tier), then the
        // generalist; the worker serving only m2 is ineligible.
        assert_eq!(
            ids(&ordered),
            vec!["idle-match", "busy-match", "generalist"]
        );

        // Unknown model: only the generalist is eligible.
        assert_eq!(ids(&registry.candidates(Some("m9"))), vec!["generalist"]);

        // No model hint: every worker by load ratio, then id.
        assert_eq!(
            ids(&registry.candidates(None)),
            vec!["generalist", "idle-match", "other", "busy-match"]
        );
    }

    #[test]
    fn load_ratio_accounts_for_capacity() {
        let registry = WorkerRegistry::new(vec![
            worker("big", &["m1"], 10),
            worker("small", &["m1"], 2),
        ]);
        // big at 5/10 = 0.5, small at 0/2 = 0.0: small first despite
        // the smaller absolute headroom.
        let _slots: Vec<WorkerSlot> = (0..5).map(|_| registry.acquire("big")).collect();
        assert_eq!(ids(&registry.candidates(Some("m1"))), vec!["small", "big"]);
        assert_eq!(registry.in_flight_count("big"), 5);
        drop(_slots);
        assert_eq!(registry.in_flight_count("big"), 0);
    }

    #[tokio::test]
    async fn try_remote_skips_failed_workers_and_notes_telemetry() {
        let registry =
            WorkerRegistry::new(vec![worker("bad", &["m1"], 4), worker("good", &["m1"], 4)]);
        let telemetry = Telemetry::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_each = seen.clone();
        let result = registry
            .try_remote(&telemetry, "m1", |worker| {
                let seen_each = seen_each.clone();
                async move {
                    seen_each.lock().unwrap().push(worker.id.clone());
                    if worker.id == "bad" {
                        Err(crate::core::error::Error::Backend("boom".to_string()))
                    } else {
                        Ok(response("remote answer"))
                    }
                }
            })
            .await;
        assert_eq!(result.unwrap().content, "remote answer");
        assert_eq!(
            *seen.lock().unwrap(),
            vec!["bad".to_string(), "good".to_string()]
        );

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.errors_per_model.get("worker:bad"), Some(&1));
        assert!(!snapshot.errors_per_model.contains_key("worker:good"));
    }

    #[tokio::test]
    async fn try_remote_returns_none_when_all_workers_fail() {
        let registry = WorkerRegistry::new(vec![worker("bad", &["m1"], 4)]);
        let telemetry = Telemetry::new();
        let result = registry
            .try_remote(&telemetry, "m1", |_| async {
                Err(crate::core::error::Error::Backend("down".to_string()))
            })
            .await;
        assert!(result.is_none());
        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.errors_per_model.get("worker:bad"), Some(&1));
    }

    #[tokio::test]
    async fn try_remote_is_none_without_workers() {
        let registry = WorkerRegistry::new(Vec::new());
        let telemetry = Telemetry::new();
        let result = registry
            .try_remote(&telemetry, "m1", |_| async { Ok(response("unreachable")) })
            .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn try_remote_stream_fails_over_to_healthy_worker() {
        use futures_util::StreamExt;
        let registry =
            WorkerRegistry::new(vec![worker("bad", &["m1"], 4), worker("good", &["m1"], 4)]);
        let telemetry = Telemetry::new();
        let result = registry
            .try_remote_stream(&telemetry, "m1", |worker| async move {
                if worker.id == "bad" {
                    Err(crate::core::error::Error::Backend("down".to_string()))
                } else {
                    let chunks = vec![Ok("hello ".to_string()), Ok("world".to_string())];
                    Ok(Box::pin(futures_util::stream::iter(chunks))
                        as crate::adapters::ChunkStream)
                }
            })
            .await;
        let mut stream = result.expect("expected a stream from the good worker");
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            text.push_str(&chunk.unwrap());
        }
        assert_eq!(text, "hello world");
        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.errors_per_model.get("worker:bad"), Some(&1));
    }

    #[tokio::test]
    async fn try_remote_stream_is_none_without_workers() {
        let registry = WorkerRegistry::new(Vec::new());
        let telemetry = Telemetry::new();
        let result = registry
            .try_remote_stream(&telemetry, "m1", |_| async {
                Ok(
                    Box::pin(futures_util::stream::iter(vec![Ok("x".to_string())]))
                        as crate::adapters::ChunkStream,
                )
            })
            .await;
        assert!(result.is_none());
    }
}
