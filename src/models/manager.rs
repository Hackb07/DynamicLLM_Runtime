use crate::adapters::{Adapter, ChunkStream};
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse};
use crate::memory::{
    EvictionListener, EvictionPolicy, MemoryMonitor, ResidencyInfo, ResidencyTracker,
};
use crate::models::lifecycle::{ModelLifecycle, ModelState};
use crate::models::registry::ModelRegistry;
use crate::telemetry::Telemetry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

/// Owns model lifecycles and their backend adapters.
///
/// Admission accounting: each local model declares `memory_required_mb`.
/// Resident models count against `local_budget_mb`, and — when the host
/// reports memory — against actual available RAM too. When a load does not
/// fit, resident models are evicted per [`EvictionPolicy`] until it fits or
/// no victim remains. Cloud models (requirement 0) always fit.
///
/// For plain HTTP adapters this is budget accounting rather than physical
/// residency: evicting a handle does not free a remote server's RAM, but it
/// keeps local admission predictable and testable. Managed adapters (a local
/// `llama-server` spawned from `gguf_path`) additionally stop their server
/// process on `unload`, which does release its RAM.
pub struct ModelManager {
    registry: Arc<RwLock<ModelRegistry>>,
    telemetry: Arc<Telemetry>,
    lifecycles: Arc<RwLock<HashMap<String, ModelLifecycle>>>,
    adapters: Arc<RwLock<HashMap<String, Arc<dyn Adapter>>>>,
    monitor: MemoryMonitor,
    eviction_policy: EvictionPolicy,
    local_budget_mb: u64,
    load_wait_timeout: Duration,
    residency: Arc<RwLock<ResidencyTracker>>,
    prefetch_permits: Arc<Semaphore>,
    prefetch_concurrency: usize,
}

impl ModelManager {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>, telemetry: Arc<Telemetry>) -> Self {
        let prefetch_concurrency = 2;
        Self {
            registry,
            telemetry,
            lifecycles: Arc::new(RwLock::new(HashMap::new())),
            adapters: Arc::new(RwLock::new(HashMap::new())),
            monitor: MemoryMonitor::new(),
            eviction_policy: EvictionPolicy::default(),
            local_budget_mb: 8192,
            load_wait_timeout: Duration::from_secs(30),
            residency: Arc::new(RwLock::new(ResidencyTracker::new())),
            prefetch_permits: Arc::new(Semaphore::new(prefetch_concurrency)),
            prefetch_concurrency,
        }
    }

    pub fn with_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    pub fn with_local_budget_mb(mut self, budget_mb: u64) -> Self {
        self.local_budget_mb = budget_mb;
        self
    }

    /// Bound for concurrent background prefetches. At most this many
    /// [`Self::prefetch`] calls load at once; the rest wait on the
    /// semaphore instead of stampeding backends.
    pub fn with_prefetch_concurrency(mut self, concurrency: usize) -> Self {
        let concurrency = concurrency.max(1);
        self.prefetch_concurrency = concurrency;
        self.prefetch_permits = Arc::new(Semaphore::new(concurrency));
        self
    }

    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction_policy
    }

    pub fn local_budget_mb(&self) -> u64 {
        self.local_budget_mb
    }

    pub fn prefetch_concurrency(&self) -> usize {
        self.prefetch_concurrency
    }

    pub fn prefetch_available_permits(&self) -> usize {
        self.prefetch_permits.available_permits()
    }

    /// Per-model residency (assumed + measured) from the eviction hooks.
    pub async fn residency_snapshot(&self) -> Vec<ResidencyInfo> {
        self.residency.read().await.snapshot()
    }

    /// Record a backend-measured footprint for a model. Currently no
    /// backend reports one, so this is primarily for tests and future
    /// adapters; admission still budgets the assumed size.
    pub async fn set_measured_memory(&self, model_id: &str, measured_mb: u64) {
        self.residency
            .write()
            .await
            .set_measured(model_id, measured_mb);
    }

    /// Install the backend adapter serving `model_id`.
    pub async fn register_adapter(&self, model_id: &str, adapter: Arc<dyn Adapter>) {
        self.adapters
            .write()
            .await
            .insert(model_id.to_string(), adapter);
        self.lifecycles
            .write()
            .await
            .entry(model_id.to_string())
            .or_insert_with(|| ModelLifecycle::new(model_id));
    }

    pub async fn adapter_for(&self, model_id: &str) -> Option<Arc<dyn Adapter>> {
        self.adapters.read().await.get(model_id).cloned()
    }

    /// Ensure `model_id` is resident and ready, loading it if needed.
    ///
    /// - `Ready` refreshes `last_used` and returns immediately.
    /// - `Loading` (by a concurrent caller) waits for completion.
    /// - Otherwise the manager admits against budget + host memory,
    ///   evicting per policy, then runs the adapter's `load()`.
    pub async fn ensure_loaded(&self, model_id: &str) -> Result<()> {
        loop {
            let state = {
                let lifecycles = self.lifecycles.read().await;
                lifecycles
                    .get(model_id)
                    .map(|lifecycle| lifecycle.state.clone())
                    .unwrap_or(ModelState::Unloaded)
            };
            match state {
                ModelState::Ready => {
                    self.touch(model_id).await;
                    return Ok(());
                }
                ModelState::Loading | ModelState::Evicting => {
                    self.wait_for_settled(model_id).await?;
                }
                ModelState::Busy => {
                    // Transient inference marker; treat as ready.
                    self.touch(model_id).await;
                    return Ok(());
                }
                ModelState::Unloaded | ModelState::Registered | ModelState::Failed => {
                    // Lost claim races return false; the loop then waits on
                    // whoever won instead of recursing.
                    if self.try_claim_and_load(model_id).await? {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Try to claim and load the model. Returns `Ok(true)` when this call
    /// performed the load, `Ok(false)` when another caller claimed it first.
    async fn try_claim_and_load(&self, model_id: &str) -> Result<bool> {
        let info = self.registry.read().await.get(model_id)?;
        let required_mb = info.memory_required_mb;

        // Claim the Loading state first so concurrent callers wait on us.
        {
            let mut lifecycles = self.lifecycles.write().await;
            let lifecycle = lifecycles
                .entry(model_id.to_string())
                .or_insert_with(|| ModelLifecycle::new(model_id));
            if matches!(
                lifecycle.state,
                ModelState::Loading | ModelState::Ready | ModelState::Evicting
            ) {
                return Ok(false);
            }
            lifecycle.state = ModelState::Loading;
            lifecycle.memory_mb = required_mb;
            lifecycle.priority = info.priority;
            lifecycle.last_error = None;
        }

        self.telemetry.record_model_load_start(model_id).await;

        let outcome = self.admit_and_init(model_id, required_mb).await;
        let mut lifecycles = self.lifecycles.write().await;
        let lifecycle = lifecycles
            .entry(model_id.to_string())
            .or_insert_with(|| ModelLifecycle::new(model_id));
        match outcome {
            Ok(()) => {
                lifecycle.state = ModelState::Ready;
                lifecycle.load_count += 1;
                lifecycle.last_used = Some(Instant::now());
                drop(lifecycles);
                self.telemetry.record_model_load_end(model_id, true).await;
                // Eviction hook: residency tracks the ASSUMED size that
                // admission budgeted; measured values arrive separately.
                self.residency.write().await.on_load(model_id, required_mb);
                Ok(true)
            }
            Err(e) => {
                lifecycle.state = ModelState::Failed;
                lifecycle.memory_mb = 0;
                lifecycle.last_error = Some(e.to_string());
                self.telemetry.record_model_load_end(model_id, false).await;
                Err(e)
            }
        }
    }

    /// Budget/host admission with eviction, then adapter initialization.
    /// Runs without holding the lifecycle lock.
    async fn admit_and_init(&self, model_id: &str, required_mb: u64) -> Result<()> {
        self.make_room(model_id, required_mb).await?;

        if let Some(available) = self.monitor.available_memory_mb() {
            if required_mb > 0 && required_mb > available {
                tracing::warn!(
                    "Host has {available}MB available but {model_id} wants {required_mb}MB; proceeding anyway"
                );
            }
        }

        let adapter = self.adapter_for(model_id).await.ok_or_else(|| {
            Error::Backend(format!("no adapter registered for model '{model_id}'"))
        })?;
        adapter.load().await?;
        Ok(())
    }

    /// Evict per policy until `required_mb` fits the local budget.
    /// Cloud models (requirement 0) always fit.
    async fn make_room(&self, model_id: &str, required_mb: u64) -> Result<()> {
        if required_mb == 0 {
            return Ok(());
        }
        if required_mb > self.local_budget_mb {
            return Err(Error::ResourceUnavailable(format!(
                "{model_id} needs {required_mb}MB, exceeding the {budget}MB local budget",
                budget = self.local_budget_mb
            )));
        }
        loop {
            let retained_mb: u64 = {
                let lifecycles = self.lifecycles.read().await;
                lifecycles
                    .values()
                    .filter(|lifecycle| {
                        lifecycle.model_id != model_id
                            && matches!(lifecycle.state, ModelState::Ready | ModelState::Busy)
                    })
                    .map(|lifecycle| lifecycle.memory_mb)
                    .sum()
            };
            if retained_mb.saturating_add(required_mb) <= self.local_budget_mb {
                return Ok(());
            }
            let victim = {
                let lifecycles = self.lifecycles.read().await;
                let residents: Vec<&ModelLifecycle> = lifecycles
                    .values()
                    .filter(|lifecycle| {
                        lifecycle.model_id != model_id && lifecycle.state == ModelState::Ready
                    })
                    .collect();
                self.eviction_policy.pick(residents)
            };
            match victim {
                Some(id) => {
                    tracing::info!(
                        "Evicting {id} to load {model_id} (policy: {:?})",
                        self.eviction_policy
                    );
                    self.unload(&id).await?;
                }
                None => {
                    return Err(Error::ResourceUnavailable(format!(
                        "cannot fit {model_id} ({required_mb}MB) in the {budget}MB local budget",
                        budget = self.local_budget_mb
                    )));
                }
            }
        }
    }

    /// Wait until a `Loading`/`Evicting` model settles, or time out.
    async fn wait_for_settled(&self, model_id: &str) -> Result<()> {
        let deadline = Instant::now() + self.load_wait_timeout;
        loop {
            let state = {
                let lifecycles = self.lifecycles.read().await;
                lifecycles
                    .get(model_id)
                    .map(|lifecycle| lifecycle.state.clone())
                    .unwrap_or(ModelState::Unloaded)
            };
            match state {
                ModelState::Ready | ModelState::Busy => return Ok(()),
                ModelState::Failed => {
                    let detail = self
                        .lifecycles
                        .read()
                        .await
                        .get(model_id)
                        .and_then(|lifecycle| lifecycle.last_error.clone())
                        .unwrap_or_else(|| "unknown error".to_string());
                    return Err(Error::ModelNotReady(format!("{model_id}: {detail}")));
                }
                ModelState::Unloaded | ModelState::Registered => return Ok(()),
                ModelState::Loading | ModelState::Evicting => {
                    if Instant::now() >= deadline {
                        return Err(Error::Timeout);
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    }

    async fn touch(&self, model_id: &str) {
        if let Some(lifecycle) = self.lifecycles.write().await.get_mut(model_id) {
            lifecycle.last_used = Some(Instant::now());
        }
    }

    pub async fn infer(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let adapter = self.adapter_for(model_id).await.ok_or_else(|| {
            Error::Backend(format!("no adapter registered for model '{model_id}'"))
        })?;
        {
            let lifecycles = self.lifecycles.read().await;
            match lifecycles.get(model_id).map(|l| &l.state) {
                Some(ModelState::Ready) | Some(ModelState::Busy) => {}
                other => {
                    return Err(Error::ModelNotReady(format!(
                        "{model_id} is not ready (state: {other:?})"
                    )))
                }
            }
        }
        let response = adapter.infer(request).await?;
        if let Some(lifecycle) = self.lifecycles.write().await.get_mut(model_id) {
            lifecycle.use_count += 1;
            lifecycle.last_used = Some(Instant::now());
        }
        Ok(response)
    }

    pub async fn infer_stream(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Result<ChunkStream> {
        let adapter = self.adapter_for(model_id).await.ok_or_else(|| {
            Error::Backend(format!("no adapter registered for model '{model_id}'"))
        })?;
        {
            let lifecycles = self.lifecycles.read().await;
            match lifecycles.get(model_id).map(|l| &l.state) {
                Some(ModelState::Ready) | Some(ModelState::Busy) => {}
                other => {
                    return Err(Error::ModelNotReady(format!(
                        "{model_id} is not ready (state: {other:?})"
                    )))
                }
            }
        }
        if let Some(lifecycle) = self.lifecycles.write().await.get_mut(model_id) {
            lifecycle.use_count += 1;
            lifecycle.last_used = Some(Instant::now());
        }
        adapter.infer_stream(request).await
    }

    pub async fn unload(&self, model_id: &str) -> Result<()> {
        {
            let mut lifecycles = self.lifecycles.write().await;
            let lifecycle = lifecycles
                .entry(model_id.to_string())
                .or_insert_with(|| ModelLifecycle::new(model_id));
            lifecycle.state = ModelState::Evicting;
        }
        // Stop backend resources (e.g. a managed `llama-server` process).
        // Best-effort: the model is evicted even if the stop fails, so a
        // wedged backend cannot pin the admission budget forever.
        if let Some(adapter) = self.adapter_for(model_id).await {
            if let Err(e) = adapter.unload().await {
                tracing::warn!("Backend stop for {model_id} failed: {e}");
            }
        }
        {
            let mut lifecycles = self.lifecycles.write().await;
            let lifecycle = lifecycles
                .entry(model_id.to_string())
                .or_insert_with(|| ModelLifecycle::new(model_id));
            lifecycle.memory_mb = 0;
            lifecycle.state = ModelState::Unloaded;
        }
        self.residency.write().await.on_evict(model_id);
        tracing::info!("Model unloaded: {model_id}");
        Ok(())
    }

    /// Bounded background prefetch: ensures `model_id` is resident,
    /// holding one prefetch permit for the whole load so at most
    /// `prefetch_concurrency` prefetches load concurrently. Already
    /// resident models return quickly (still briefly holding a permit).
    /// Spawn it (`tokio::spawn`) for fire-and-forget warmup; await it
    /// directly when the caller needs the model warm before proceeding.
    pub async fn prefetch(&self, model_id: &str) -> Result<()> {
        let _permit = self
            .prefetch_permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Cancelled)?;
        self.ensure_loaded(model_id).await
    }

    /// Fire-and-forget background prefetch on an `Arc` manager. Returns
    /// the join handle so callers can await or detach; load failures are
    /// returned from the handle, never panicked.
    pub fn spawn_prefetch(
        self: &Arc<Self>,
        model_id: impl Into<String>,
    ) -> tokio::task::JoinHandle<Result<()>> {
        let manager = Arc::clone(self);
        let model_id = model_id.into();
        tokio::spawn(async move { manager.prefetch(&model_id).await })
    }

    /// Prefetch a warm set in the background, one task per model,
    /// bounded by the shared prefetch semaphore.
    pub fn spawn_warmup(
        self: &Arc<Self>,
        model_ids: &[String],
    ) -> Vec<tokio::task::JoinHandle<Result<()>>> {
        model_ids
            .iter()
            .map(|id| self.spawn_prefetch(id.clone()))
            .collect()
    }

    /// Ids of currently resident models.
    pub async fn loaded_models(&self) -> Vec<String> {
        let lifecycles = self.lifecycles.read().await;
        let mut loaded: Vec<String> = lifecycles
            .values()
            .filter(|lifecycle| {
                matches!(
                    lifecycle.state,
                    ModelState::Ready | ModelState::Busy | ModelState::Loading
                )
            })
            .map(|lifecycle| lifecycle.model_id.clone())
            .collect();
        loaded.sort();
        loaded
    }

    /// Resident megabytes currently counted against the local budget.
    pub async fn retained_mb(&self) -> u64 {
        self.lifecycles
            .read()
            .await
            .values()
            .filter(|lifecycle| {
                matches!(
                    lifecycle.state,
                    ModelState::Ready | ModelState::Busy | ModelState::Loading
                )
            })
            .map(|lifecycle| lifecycle.memory_mb)
            .sum()
    }

    pub async fn lifecycle_snapshot(&self) -> HashMap<String, ModelState> {
        self.lifecycles
            .read()
            .await
            .iter()
            .map(|(id, lifecycle)| (id.clone(), lifecycle.state.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Usage;
    use async_trait::async_trait;

    struct StubAdapter {
        healthy: bool,
        load_delay_ms: u64,
    }

    #[async_trait]
    impl Adapter for StubAdapter {
        fn name(&self) -> &str {
            "stub"
        }

        fn backend(&self) -> &str {
            "stub"
        }

        async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
            Ok(InferenceResponse {
                request_id: request.request_id.clone().unwrap_or_default(),
                model: "stub".to_string(),
                content: "stubbed".to_string(),
                finish_reason: Some("stop".to_string()),
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                }),
                tool_calls: None,
                metadata: None,
            })
        }

        async fn load(&self) -> Result<()> {
            if self.load_delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.load_delay_ms)).await;
            }
            if self.healthy {
                Ok(())
            } else {
                Err(Error::ResourceUnavailable("stub backend down".to_string()))
            }
        }

        async fn health_check(&self) -> Result<bool> {
            Ok(self.healthy)
        }
    }

    fn test_manager() -> (ModelManager, Arc<RwLock<ModelRegistry>>) {
        let registry = Arc::new(RwLock::new(ModelRegistry::new()));
        let telemetry = Arc::new(Telemetry::new());
        let manager = ModelManager::new(registry.clone(), telemetry).with_local_budget_mb(4096);
        (manager, registry)
    }

    #[tokio::test]
    async fn cloud_model_loads_without_budget() {
        let (manager, _) = test_manager();
        manager
            .register_adapter(
                "gpt-3.5-turbo",
                Arc::new(StubAdapter {
                    healthy: true,
                    load_delay_ms: 0,
                }),
            )
            .await;
        manager.ensure_loaded("gpt-3.5-turbo").await.unwrap();
        assert_eq!(manager.retained_mb().await, 0);
        assert_eq!(
            manager.loaded_models().await,
            vec!["gpt-3.5-turbo".to_string()]
        );
    }

    #[tokio::test]
    async fn load_failure_marks_model_failed() {
        let (manager, _) = test_manager();
        manager
            .register_adapter(
                "llama-3.2-3b",
                Arc::new(StubAdapter {
                    healthy: false,
                    load_delay_ms: 0,
                }),
            )
            .await;
        let err = manager.ensure_loaded("llama-3.2-3b").await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)));
        let snapshot = manager.lifecycle_snapshot().await;
        assert_eq!(snapshot.get("llama-3.2-3b"), Some(&ModelState::Failed));
    }

    #[tokio::test]
    async fn lru_victim_is_evicted_to_fit_budget() {
        let (manager, registry) = test_manager();
        // Two 3072MB models under a 4096MB budget: loading the second
        // must evict the first.
        for id in ["local-a", "local-b"] {
            registry
                .write()
                .await
                .register(crate::models::registry::ModelInfo {
                    id: id.to_string(),
                    name: id.to_string(),
                    backend: "stub".to_string(),
                    capabilities: crate::models::capabilities::ModelCapabilities {
                        supports_tools: false,
                        supports_streaming: false,
                        supports_structured_output: false,
                        task_affinity: vec![],
                    },
                    context_limit: 1024,
                    memory_required_mb: 3072,
                    quantization: None,
                    enabled: true,
                    gguf_path: None,
                    local_port: None,
                    priority: 0,
                });
            manager
                .register_adapter(
                    id,
                    Arc::new(StubAdapter {
                        healthy: true,
                        load_delay_ms: 0,
                    }),
                )
                .await;
        }
        manager.ensure_loaded("local-a").await.unwrap();
        manager.ensure_loaded("local-b").await.unwrap();

        let snapshot = manager.lifecycle_snapshot().await;
        assert_eq!(snapshot.get("local-a"), Some(&ModelState::Unloaded));
        assert_eq!(snapshot.get("local-b"), Some(&ModelState::Ready));
        assert_eq!(manager.retained_mb().await, 3072);
    }

    #[tokio::test]
    async fn oversized_model_is_rejected_without_evicting() {
        let (manager, registry) = test_manager();
        registry
            .write()
            .await
            .register(crate::models::registry::ModelInfo {
                id: "huge".to_string(),
                name: "huge".to_string(),
                backend: "stub".to_string(),
                capabilities: crate::models::capabilities::ModelCapabilities {
                    supports_tools: false,
                    supports_streaming: false,
                    supports_structured_output: false,
                    task_affinity: vec![],
                },
                context_limit: 1024,
                memory_required_mb: 1 << 30,
                quantization: None,
                enabled: true,
                gguf_path: None,
                local_port: None,
                priority: 0,
            });
        manager
            .register_adapter(
                "huge",
                Arc::new(StubAdapter {
                    healthy: true,
                    load_delay_ms: 0,
                }),
            )
            .await;
        let err = manager.ensure_loaded("huge").await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)));
    }

    #[tokio::test]
    async fn concurrent_loaders_wait_on_one_load() {
        let (manager, _) = test_manager();
        let manager = Arc::new(manager);
        manager
            .register_adapter(
                "llama-3.2-3b",
                Arc::new(StubAdapter {
                    healthy: true,
                    load_delay_ms: 200,
                }),
            )
            .await;
        let first = {
            let manager = manager.clone();
            tokio::spawn(async move { manager.ensure_loaded("llama-3.2-3b").await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        let second = manager.ensure_loaded("llama-3.2-3b").await;
        let first = first.await.unwrap();
        assert!(first.is_ok());
        assert!(second.is_ok());
        let snapshot = manager.lifecycle_snapshot().await;
        assert_eq!(snapshot.get("llama-3.2-3b"), Some(&ModelState::Ready));
    }

    #[tokio::test]
    async fn infer_requires_ready_state() {
        let (manager, _) = test_manager();
        manager
            .register_adapter(
                "gpt-3.5-turbo",
                Arc::new(StubAdapter {
                    healthy: true,
                    load_delay_ms: 0,
                }),
            )
            .await;
        let request = InferenceRequest {
            request_id: None,
            prompt: "hi".to_string(),
            model: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            metadata: None,
        };
        let err = manager.infer("gpt-3.5-turbo", &request).await.unwrap_err();
        assert!(matches!(err, Error::ModelNotReady(_)));
    }

    struct UnloadProbe {
        stopped: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl Adapter for UnloadProbe {
        fn name(&self) -> &str {
            "probe"
        }

        fn backend(&self) -> &str {
            "probe"
        }

        async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
            Ok(InferenceResponse {
                request_id: request.request_id.clone().unwrap_or_default(),
                model: "probe".to_string(),
                content: String::new(),
                finish_reason: None,
                usage: None,
                tool_calls: None,
                metadata: None,
            })
        }

        async fn unload(&self) -> Result<()> {
            self.stopped
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        async fn health_check(&self) -> Result<bool> {
            Ok(true)
        }
    }

    #[tokio::test]
    async fn unload_stops_the_backend_adapter() {
        let (manager, registry) = test_manager();
        registry
            .write()
            .await
            .register(crate::models::registry::ModelInfo {
                id: "managed".to_string(),
                name: "managed".to_string(),
                backend: "probe".to_string(),
                capabilities: crate::models::capabilities::ModelCapabilities {
                    supports_tools: false,
                    supports_streaming: false,
                    supports_structured_output: false,
                    task_affinity: vec![],
                },
                context_limit: 1024,
                memory_required_mb: 0,
                quantization: None,
                enabled: true,
                gguf_path: None,
                local_port: None,
                priority: 0,
            });
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        manager
            .register_adapter(
                "managed",
                Arc::new(UnloadProbe {
                    stopped: stopped.clone(),
                }),
            )
            .await;
        manager.ensure_loaded("managed").await.unwrap();
        manager.unload("managed").await.unwrap();
        assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
        let snapshot = manager.lifecycle_snapshot().await;
        assert_eq!(snapshot.get("managed"), Some(&ModelState::Unloaded));
    }

    #[tokio::test]
    async fn priority_aware_evicts_lowest_priority_first() {
        let registry = Arc::new(RwLock::new(ModelRegistry::new()));
        let telemetry = Arc::new(Telemetry::new());
        let manager = ModelManager::new(registry.clone(), telemetry)
            .with_policy(EvictionPolicy::PriorityAware)
            .with_local_budget_mb(8192);
        // Three 3072MB models under an 8192MB budget: the third load must
        // evict exactly one resident. Priorities come from `ModelInfo`.
        for (id, priority) in [("old-hi", 9), ("mid-lo", 1), ("newbie", 5)] {
            registry
                .write()
                .await
                .register(crate::models::registry::ModelInfo {
                    id: id.to_string(),
                    name: id.to_string(),
                    backend: "stub".to_string(),
                    capabilities: crate::models::capabilities::ModelCapabilities {
                        supports_tools: false,
                        supports_streaming: false,
                        supports_structured_output: false,
                        task_affinity: vec![],
                    },
                    context_limit: 1024,
                    memory_required_mb: 3072,
                    quantization: None,
                    enabled: true,
                    gguf_path: None,
                    local_port: None,
                    priority,
                });
            manager
                .register_adapter(
                    id,
                    Arc::new(StubAdapter {
                        healthy: true,
                        load_delay_ms: 0,
                    }),
                )
                .await;
        }
        manager.ensure_loaded("old-hi").await.unwrap();
        manager.ensure_loaded("mid-lo").await.unwrap();
        manager.ensure_loaded("newbie").await.unwrap();

        // LRU would evict "old-hi" (loaded first); priority-aware eviction
        // must evict "mid-lo" (priority 1) instead.
        let snapshot = manager.lifecycle_snapshot().await;
        assert_eq!(snapshot.get("old-hi"), Some(&ModelState::Ready));
        assert_eq!(snapshot.get("mid-lo"), Some(&ModelState::Unloaded));
        assert_eq!(snapshot.get("newbie"), Some(&ModelState::Ready));
    }
}
