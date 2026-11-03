use crate::adapters::{
    Adapter, ChunkStream, LlamaCppAdapter, LocalEngineAdapter, ManagedServerConfig, OpenAIAdapter,
    WorkerClient,
};
use crate::agents::{
    AgentExecutor, AgentRequest, AgentResponse, SessionManager, ToolDefinition,
    DEFAULT_PER_AGENT_CONCURRENCY,
};
use crate::config::RuntimeConfig;
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse};
use crate::memory::{EvictionPolicy, MemoryMonitor, ResidencyInfo};
use crate::models::manager::ModelManager;
use crate::models::registry::{ModelInfo, ModelRegistry};
use crate::routing::router::ModelRouter;
use crate::routing::strategy::{RoutingStrategy, RoutingWeights};
use crate::scheduling::predictor::NextModelPredictor;
use crate::scheduling::scheduler::Scheduler;
use crate::scheduling::workers::WorkerRegistry;
use crate::telemetry::Telemetry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};

/// Central execution runtime: validates requests, routes to a model,
/// enforces admission + concurrency limits, runs inference with failover,
/// and records telemetry.
pub struct Runtime {
    registry: Arc<RwLock<ModelRegistry>>,
    manager: Arc<ModelManager>,
    router: Arc<ModelRouter>,
    telemetry: Arc<Telemetry>,
    scheduler: Arc<Scheduler>,
    agent_executor: AgentExecutor,
    agent_scheduler: Arc<Scheduler>,
    sessions: SessionManager,
    monitor: MemoryMonitor,
    predictor: Arc<Mutex<NextModelPredictor>>,
    workers: WorkerRegistry,
    worker_clients: HashMap<String, WorkerClient>,
    config: RuntimeConfig,
}

impl Runtime {
    /// Build a runtime from config. `models` replaces the built-in defaults
    /// when non-empty (i.e. a config file was provided); otherwise the
    /// registry keeps its defaults.
    ///
    /// Async because adapter installation takes async locks; call it from
    /// `main`, request handlers, or tests, all of which are async.
    pub async fn with_config(config: RuntimeConfig, models: Vec<ModelInfo>) -> Result<Self> {
        let telemetry = Arc::new(Telemetry::new());

        let mut registry = ModelRegistry::new();
        if !models.is_empty() {
            registry.clear();
            for model in models {
                registry.register(model);
            }
        }
        let registry = Arc::new(RwLock::new(registry));

        let manager = Arc::new(
            ModelManager::new(registry.clone(), telemetry.clone())
                .with_policy(EvictionPolicy::parse(&config.eviction_policy))
                .with_local_budget_mb(config.local_model_budget_mb)
                .with_prefetch_concurrency(config.prefetch_concurrency),
        );
        // Adapter installation only builds in-process HTTP handles;
        // it never touches the network.
        install_adapters(&manager, &registry, &config).await;

        let strategy = RoutingStrategy::parse(&config.routing_strategy);
        let weights = RoutingWeights::new(
            config.routing_latency_weight,
            config.routing_success_weight,
            config.routing_affinity_bonus,
        );
        let mut router = ModelRouter::new(registry.clone())
            .with_strategy(strategy)
            .with_telemetry(telemetry.clone())
            .with_weights(weights);
        if strategy == RoutingStrategy::Fixed {
            if let Some(model) = &config.fixed_model {
                router = router.with_fixed_model(model.clone());
            }
        }
        let router = Arc::new(router);

        let scheduler = Arc::new(Scheduler::new(config.max_concurrent_requests));
        // Agent runs are long-lived multi-step loops, so they get their own
        // concurrency gate instead of sharing the per-request one.
        let agent_scheduler = Arc::new(Scheduler::new(config.max_concurrent_requests));

        let predictor = Arc::new(Mutex::new(NextModelPredictor::with_history_limit(
            config.predictor_history_limit,
        )));

        // Opt-in remote execution: build one cached HTTP client per
        // configured worker. Empty by default; client-build failures are
        // logged and that worker is skipped (local fallback still works).
        let workers = WorkerRegistry::from_config(&config.workers);
        let worker_timeout = Duration::from_secs(config.request_timeout_secs);
        let mut worker_clients = HashMap::new();
        for worker in workers.workers() {
            match WorkerClient::new(
                &worker.id,
                &worker.addr,
                worker_timeout,
                config.max_retries,
                config.retry_backoff_ms,
            ) {
                Ok(client) => {
                    worker_clients.insert(worker.id.clone(), client);
                }
                Err(e) => {
                    tracing::error!("Cannot build worker client for {}: {e}", worker.id);
                }
            }
        }

        let sessions = SessionManager::new(
            config.max_concurrent_requests,
            DEFAULT_PER_AGENT_CONCURRENCY,
        );

        // Background warmup: fire-and-forget prefetches for the configured
        // warm set, bounded by the manager's prefetch semaphore. Handles
        // are detached (dropping a JoinHandle leaves the task running);
        // prefetch failures only log inside the task, and the next demand
        // load retries.
        if !config.prefetch_warm_models.is_empty() {
            let warm: Vec<String> = config
                .prefetch_warm_models
                .iter()
                .filter(|id| !id.trim().is_empty())
                .cloned()
                .collect();
            let _detached = manager.spawn_warmup(&warm);
            tracing::info!("Warming {} model(s) in the background", warm.len());
        }

        Ok(Self {
            registry,
            manager,
            router,
            telemetry,
            scheduler,
            agent_executor: AgentExecutor::new(),
            agent_scheduler,
            sessions,
            monitor: MemoryMonitor::new(),
            predictor,
            workers,
            worker_clients,
            config,
        })
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub async fn infer(&self, mut request: InferenceRequest) -> Result<InferenceResponse> {
        validate_request(&request)?;
        // Concurrency gate: hold the permit for the whole request.
        let _permit = self.scheduler.acquire().await?;

        let request_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        request.request_id = Some(request_id.clone());

        self.telemetry.record_request_start(&request_id).await;
        // Snapshot the predictor's guess before serving so the outcome
        // (predicted vs actually demanded) scores the prefetch hit rate.
        let predicted = self.predictor.lock().await.predict();

        let candidates = match self.router.route_candidates(&request).await {
            Ok(candidates) => candidates,
            Err(e) => {
                self.telemetry
                    .record_request_end(&request_id, "none", false)
                    .await;
                return Err(e);
            }
        };
        let attempts: Vec<String> = candidates
            .into_iter()
            .take(self.config.max_fallback_models.max(1))
            .collect();
        self.telemetry.record_routing(&request_id, &attempts).await;

        let mut last_error = Error::Routing("No available models".to_string());
        for model_id in &attempts {
            match self.try_model(model_id, &request).await {
                Ok(response) => {
                    self.telemetry
                        .record_request_end(&request_id, model_id, true)
                        .await;
                    self.observe_served_model(predicted.as_deref(), model_id)
                        .await;
                    return Ok(response);
                }
                Err(e) => {
                    self.telemetry
                        .record_error(&request_id, model_id, &e.to_string())
                        .await;
                    tracing::warn!("{model_id} failed for {request_id}: {e}");
                    if is_failover_error(&e) && attempts.len() > 1 {
                        last_error = e;
                        continue;
                    }
                    self.telemetry
                        .record_request_end(&request_id, model_id, false)
                        .await;
                    return Err(e);
                }
            }
        }
        self.telemetry
            .record_request_end(
                &request_id,
                attempts.first().map(String::as_str).unwrap_or("none"),
                false,
            )
            .await;
        Err(last_error)
    }

    async fn try_model(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        // Stamp the serving model so backends (and the response) agree
        // on which model actually handled the request.
        let mut request = request.clone();
        request.model = Some(model_id.to_string());
        // Opt-in remote execution: with no workers configured this
        // returns None immediately and the local path below runs exactly
        // as before. Worker failures fall through to local execution
        // with a `worker:<id>` telemetry note each.
        if let Some(response) = self.try_remote_workers(model_id, &request).await {
            return Ok(response);
        }
        self.manager.ensure_loaded(model_id).await?;
        let timeout = Duration::from_secs(self.config.request_timeout_secs);
        tokio::time::timeout(timeout, self.manager.infer(model_id, &request))
            .await
            .map_err(|_| Error::Timeout)?
    }

    /// Attempt remote execution on configured workers in placement order.
    /// `None` means "no workers" or "all workers failed": the caller
    /// falls back to local execution.
    async fn try_remote_workers(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Option<InferenceResponse> {
        if self.workers.is_empty() {
            return None;
        }
        self.workers
            .try_remote(&self.telemetry, model_id, |worker| {
                let owned = request.clone();
                let model = model_id.to_string();
                let client = self.worker_clients.get(&worker.id).cloned();
                async move {
                    match client {
                        Some(client) => client.infer(&model, &owned).await,
                        None => Err(Error::Backend(format!(
                            "worker {} has no client",
                            worker.id
                        ))),
                    }
                }
            })
            .await
    }

    /// Streaming twin of [`Self::try_remote_workers`]: attempts remote
    /// stream startup on configured workers in placement order. `None`
    /// means "no workers" or "all workers failed to start a stream": the
    /// caller falls back to local execution.
    async fn try_remote_stream_workers(
        &self,
        model_id: &str,
        request: &InferenceRequest,
    ) -> Option<ChunkStream> {
        if self.workers.is_empty() {
            return None;
        }
        self.workers
            .try_remote_stream(&self.telemetry, model_id, |worker| {
                let owned = request.clone();
                let model = model_id.to_string();
                let client = self.worker_clients.get(&worker.id).cloned();
                async move {
                    match client {
                        Some(client) => client.infer_stream(&model, &owned).await,
                        None => Err(Error::Backend(format!(
                            "worker {} has no client",
                            worker.id
                        ))),
                    }
                }
            })
            .await
    }

    /// Record the served model in the frequency predictor, score the
    /// pre-snapshot prediction into the prefetch hit rate, and — when
    /// `predictive_prefetch_enabled` — prefetch the newly predicted next
    /// model in the background. Only successful serves are observed;
    /// failures leave the demand history unchanged.
    async fn observe_served_model(&self, predicted: Option<&str>, actual: &str) {
        let next = {
            let mut predictor = self.predictor.lock().await;
            // The very first request has no prediction to score; only
            // subsequent requests count toward the hit rate.
            if let Some(guess) = predicted {
                let hit = predictor.record_outcome(Some(guess), actual);
                self.telemetry.record_prefetch_outcome(hit).await;
            }
            predictor.observe(actual);
            predictor.predict()
        };
        if self.config.predictive_prefetch_enabled {
            if let Some(next) = next.filter(|id| id != actual) {
                let _detached = self.manager.spawn_prefetch(next);
            }
        }
    }

    /// Streaming inference. Uses the primary candidate only: failing over
    /// mid-stream would duplicate or reorder tokens. Remote workers are
    /// attempted first at stream *startup* (with failover across workers);
    /// once chunks flow, the stream stays on its winner.
    pub async fn infer_stream(&self, mut request: InferenceRequest) -> Result<ChunkStream> {
        validate_request(&request)?;
        // The permit cannot be held across the returned stream (it is not
        // part of the stream), so streaming admission is check-and-release:
        // it still bounds how fast streams can start.
        drop(self.scheduler.acquire().await?);

        let request_id = request
            .request_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        request.request_id = Some(request_id.clone());

        self.telemetry.record_request_start(&request_id).await;
        let predicted = self.predictor.lock().await.predict();
        let model_id = match self.router.route(&request).await {
            Ok(model_id) => model_id,
            Err(e) => {
                self.telemetry
                    .record_request_end(&request_id, "none", false)
                    .await;
                return Err(e);
            }
        };
        self.telemetry
            .record_routing(&request_id, std::slice::from_ref(&model_id))
            .await;
        request.model = Some(model_id.clone());
        if let Some(stream) = self.try_remote_stream_workers(&model_id, &request).await {
            self.telemetry
                .record_request_end(&request_id, &model_id, true)
                .await;
            self.observe_served_model(predicted.as_deref(), &model_id)
                .await;
            return Ok(stream);
        }
        if let Err(e) = self.manager.ensure_loaded(&model_id).await {
            self.telemetry
                .record_request_end(&request_id, &model_id, false)
                .await;
            return Err(e);
        }
        match self.manager.infer_stream(&model_id, &request).await {
            Ok(stream) => {
                self.telemetry
                    .record_request_end(&request_id, &model_id, true)
                    .await;
                self.observe_served_model(predicted.as_deref(), &model_id)
                    .await;
                Ok(stream)
            }
            Err(e) => {
                self.telemetry
                    .record_request_end(&request_id, &model_id, false)
                    .await;
                Err(e)
            }
        }
    }

    /// Run one agentic task: bounded concurrency, per-run telemetry.
    /// Terminal agent failures arrive as `Ok(response)` with `state` set
    /// (audit preserved); only malformed requests return `Err`.
    pub async fn run_agent(&self, request: AgentRequest) -> Result<AgentResponse> {
        let _permit = self.agent_scheduler.acquire().await?;

        let request_id = format!("agent-{}", uuid::Uuid::new_v4());
        self.telemetry.record_request_start(&request_id).await;
        let model_label = request.model.clone().unwrap_or_else(|| "agent".to_string());

        match self.agent_executor.run(self, request).await {
            Ok(response) => {
                if !response.is_success() {
                    if let Some(error) = response.error.as_deref() {
                        self.telemetry
                            .record_error(&request_id, &model_label, error)
                            .await;
                    }
                }
                self.telemetry
                    .record_request_end(&request_id, &model_label, response.is_success())
                    .await;
                Ok(response)
            }
            Err(e) => {
                self.telemetry
                    .record_error(&request_id, &model_label, &e.to_string())
                    .await;
                self.telemetry
                    .record_request_end(&request_id, &model_label, false)
                    .await;
                Err(e)
            }
        }
    }

    /// Tool definitions the agent executor can run (deny-by-default at
    /// execution time: a run must also allowlist each tool it may use).
    pub fn list_agent_tools(&self) -> Vec<ToolDefinition> {
        self.agent_executor.registry().list().clone()
    }

    /// The in-memory multi-agent session manager.
    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    /// Create an empty agent session sharing one tool policy.
    pub async fn create_agent_session(&self, allowed_tools: Vec<String>) -> String {
        self.sessions
            .create_session_with_policy(allowed_tools)
            .await
    }

    /// Run one agentic task as `agent_id` inside `session_id`. The session
    /// must exist (the agent is auto-created); the run's effective tool
    /// allowlist is the request intersected with the session's shared
    /// policy. Concurrency is bounded per agent and globally by the
    /// session manager, on top of the usual agent gate. Terminal state
    /// and transcript are mirrored into the session's agent record.
    pub async fn run_agent_in_session(
        &self,
        session_id: &str,
        agent_id: &str,
        request: AgentRequest,
    ) -> Result<AgentResponse> {
        let effective = self
            .sessions
            .effective_tools(session_id, &request.allowed_tools)
            .await?;
        self.sessions.spawn_agent(session_id, agent_id).await?;
        let _permit = self.sessions.acquire(session_id, agent_id).await?;
        let scoped = AgentRequest {
            allowed_tools: effective,
            ..request
        };
        let response = self.run_agent(scoped).await?;
        // The run already completed: mirror best-effort so a session
        // removed mid-run cannot fail the response.
        let _ = self
            .sessions
            .set_agent_state(session_id, agent_id, response.state)
            .await;
        let _ = self
            .sessions
            .record_agent_steps(session_id, agent_id, response.steps_used)
            .await;
        for line in &response.transcript {
            let _ = self
                .sessions
                .push_agent_transcript(session_id, agent_id, line.clone())
                .await;
        }
        Ok(response)
    }

    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.registry.read().await.list_models()
    }

    /// Predictor's current next-model guess, if any history exists.
    pub async fn predict_next_model(&self) -> Option<String> {
        self.predictor.lock().await.predict()
    }

    /// Predictor hit rate (None until at least one scored prediction).
    pub async fn predictor_hit_rate(&self) -> Option<f64> {
        self.predictor.lock().await.hit_rate()
    }

    /// Bounded prefetch of one model (awaits the load).
    pub async fn prefetch_model(&self, model_id: &str) -> Result<()> {
        self.manager.prefetch(model_id).await
    }

    /// Fire-and-forget background prefetch of one model.
    pub fn spawn_prefetch(&self, model_id: String) -> tokio::task::JoinHandle<Result<()>> {
        self.manager.spawn_prefetch(model_id)
    }

    /// Per-model residency (assumed + measured) from eviction hooks.
    pub async fn residency_snapshot(&self) -> Vec<ResidencyInfo> {
        self.manager.residency_snapshot().await
    }

    pub async fn get_status(&self) -> serde_json::Value {
        let (predicted, hits, total) = {
            let predictor = self.predictor.lock().await;
            (
                predictor.predict(),
                predictor.hits(),
                predictor.total_predictions(),
            )
        };
        serde_json::json!({
            "status": "running",
            "models": self.list_models().await.len(),
            "loaded_models": self.manager.loaded_models().await,
            "lifecycle": self.manager.lifecycle_snapshot().await,
            "memory": self.monitor.snapshot(),
            "residency": self.manager.residency_snapshot().await,
            "scheduler": {
                "available_permits": self.scheduler.available_permits(),
                "max_concurrent": self.config.max_concurrent_requests,
            },
            "agent": {
                "available_permits": self.agent_scheduler.available_permits(),
                "max_concurrent": self.config.max_concurrent_requests,
                "tools": self.agent_executor.registry().list().len(),
                "sessions": self.sessions.session_count().await,
                "session_permits": self.sessions.available_permits(),
            },
            "distributed": {
                "enabled": !self.workers.is_empty(),
                "workers": self.workers.workers().iter().map(|worker| {
                    serde_json::json!({
                        "id": worker.id,
                        "addr": worker.addr,
                        "models": worker.models,
                        "capacity": worker.capacity,
                        "in_flight": self.workers.in_flight_count(&worker.id),
                    })
                }).collect::<Vec<_>>(),
            },
            "predictor": {
                "next": predicted,
                "hits": hits,
                "total": total,
                "hit_rate": self.telemetry.prefetch_hit_rate().await,
                "enabled": self.config.predictive_prefetch_enabled,
            },
            "prefetch": {
                "concurrency": self.manager.prefetch_concurrency(),
                "available_permits": self.manager.prefetch_available_permits(),
            },
            "telemetry": self.telemetry.snapshot().await,
        })
    }
}

/// Build one adapter per registered model, keyed by its `backend` value.
/// Unknown backends are logged and left without an adapter so requests for
/// them fail with a clear error instead of a mock response.
async fn install_adapters(
    manager: &ModelManager,
    registry: &Arc<RwLock<ModelRegistry>>,
    config: &RuntimeConfig,
) {
    let models = registry.read().await.list_models();
    let timeout = Duration::from_secs(config.request_timeout_secs);
    let mut managed_ports: Vec<u16> = Vec::new();
    for model in models {
        let adapter: Option<Arc<dyn Adapter>> = match model.backend.as_str() {
            "openai" => match OpenAIAdapter::from_config(
                &config.openai_base_url,
                config.openai_api_key(),
                &config.openai_default_model,
                timeout,
                config.max_retries,
                config.retry_backoff_ms,
            ) {
                Ok(adapter) => Some(Arc::new(adapter)),
                Err(e) => {
                    tracing::error!("Cannot build OpenAI adapter for {}: {e}", model.id);
                    None
                }
            },
            "llama.cpp" | "llamacpp" | "llama_cpp" => {
                build_llamacpp_adapter(&model, config, timeout, &mut managed_ports)
            }
            // In-process engine: GGUF weights are memory-mapped into this
            // process (SSD-backed) instead of served over HTTP. Requires
            // `gguf_path`; context size reuses `llamacpp_ctx_size`.
            "local" => match model.gguf_path.as_deref() {
                Some(gguf_path) => Some(Arc::new(
                    LocalEngineAdapter::new(gguf_path, &model.id, config.llamacpp_ctx_size)
                        .with_gpu_layers(config.n_gpu_layers),
                )),
                None => {
                    tracing::error!(
                        "Model {} uses the 'local' backend without gguf_path; no adapter installed",
                        model.id
                    );
                    None
                }
            },
            other => {
                tracing::warn!(
                    "Model {} uses unknown backend '{other}'; no adapter installed",
                    model.id
                );
                None
            }
        };
        if let Some(adapter) = adapter {
            manager.register_adapter(&model.id, adapter).await;
        }
    }
}

/// Build the adapter for one `llama.cpp` model. Models with `gguf_path`
/// get a managed adapter that spawns its own `llama-server` on its own
/// port; models without one keep the legacy external-server behavior.
fn build_llamacpp_adapter(
    model: &ModelInfo,
    config: &RuntimeConfig,
    timeout: Duration,
    managed_ports: &mut Vec<u16>,
) -> Option<Arc<dyn Adapter>> {
    if let Some(gguf_path) = model.gguf_path.as_deref() {
        let port = model
            .local_port
            .unwrap_or_else(|| default_llamacpp_port(config));
        if managed_ports.contains(&port) {
            tracing::error!(
                "Model {} wants managed port {port}, already used by another model; \
                 give each managed model its own local_port",
                model.id
            );
            return None;
        }
        managed_ports.push(port);
        let managed = ManagedServerConfig {
            binary: config.llamacpp_binary.clone(),
            gguf_path: gguf_path.to_string(),
            port,
            ctx_size: config.llamacpp_ctx_size,
            n_gpu_layers: config.n_gpu_layers,
            extra_args: config.llamacpp_extra_args.clone(),
        };
        // Local servers answer slowly; more than one retry would double
        // tail latency for little gain, so retries stay clamped at one.
        return match LlamaCppAdapter::managed(
            managed,
            &model.id,
            timeout,
            config.max_retries.min(1),
            config.retry_backoff_ms,
        ) {
            Ok(adapter) => Some(Arc::new(adapter)),
            Err(e) => {
                tracing::error!(
                    "Cannot build managed llama.cpp adapter for {}: {e}",
                    model.id
                );
                None
            }
        };
    }
    match LlamaCppAdapter::from_config(
        &config.llamacpp_base_url,
        &config.llamacpp_default_model,
        timeout,
        config.max_retries.min(1),
        config.retry_backoff_ms,
    ) {
        Ok(adapter) => Some(Arc::new(adapter)),
        Err(e) => {
            tracing::error!("Cannot build llama.cpp adapter for {}: {e}", model.id);
            None
        }
    }
}

/// Default port for a managed server: the port parsed from
/// `llamacpp_base_url`, falling back to 8081.
fn default_llamacpp_port(config: &RuntimeConfig) -> u16 {
    config
        .llamacpp_base_url
        .rsplit(':')
        .next()
        .and_then(|port| port.trim_end_matches('/').parse::<u16>().ok())
        .unwrap_or(8081)
}

/// Errors worth failing over to the next candidate for. Validation-style
/// errors return immediately since another model would fail the same way.
fn is_failover_error(error: &Error) -> bool {
    matches!(
        error,
        Error::Backend(_)
            | Error::Timeout
            | Error::ModelNotReady(_)
            | Error::ResourceUnavailable(_)
    )
}

/// Shared request validation for all entry points (`infer` and
/// `infer_stream`; the `stream` flag itself is only a routing hint — the
/// endpoint decides the wire format, so `POST /v1/infer` with
/// `stream: true` still returns one JSON object).
fn validate_request(request: &InferenceRequest) -> Result<()> {
    if request.prompt.trim().is_empty() {
        return Err(Error::Validation("prompt must not be empty".to_string()));
    }
    if request.max_tokens == Some(0) {
        return Err(Error::Validation("max_tokens must be positive".to_string()));
    }
    if let Some(temperature) = request.temperature {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(Error::Validation(
                "temperature must be within 0.0 and 2.0".to_string(),
            ));
        }
    }
    if let Some(top_p) = request.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(Error::Validation(
                "top_p must be within 0.0 and 1.0".to_string(),
            ));
        }
    }
    if let Some(stop) = request.stop.as_ref() {
        if stop.iter().any(|s| s.is_empty()) {
            return Err(Error::Validation(
                "stop sequences must not be empty".to_string(),
            ));
        }
    }
    if let Some(tools) = request.tools.as_ref() {
        if tools.iter().any(|t| t.name.trim().is_empty()) {
            return Err(Error::Validation(
                "tool names must not be empty".to_string(),
            ));
        }
    }
    if let Some(request_id) = request.request_id.as_ref() {
        if request_id.trim().is_empty() {
            return Err(Error::Validation(
                "request_id must not be blank".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: &str) -> InferenceRequest {
        InferenceRequest {
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

    #[test]
    fn validation_rules() {
        assert!(validate_request(&request("hello")).is_ok());
        assert!(validate_request(&request("   ")).is_err());
        let mut bad = request("hello");
        bad.max_tokens = Some(0);
        assert!(validate_request(&bad).is_err());
        let mut bad = request("hello");
        bad.temperature = Some(2.5);
        assert!(validate_request(&bad).is_err());
        let mut bad = request("hello");
        bad.top_p = Some(-0.1);
        assert!(validate_request(&bad).is_err());
        let mut bad = request("hello");
        bad.stop = Some(vec![String::new()]);
        assert!(validate_request(&bad).is_err());
        let mut bad = request("hello");
        bad.request_id = Some("   ".to_string());
        assert!(validate_request(&bad).is_err());
        // An empty tools vec is valid (treated as no tool requirement).
        let mut ok = request("hello");
        ok.tools = Some(Vec::new());
        assert!(validate_request(&ok).is_ok());
    }

    #[test]
    fn failover_classification() {
        assert!(is_failover_error(&Error::Backend("x".to_string())));
        assert!(is_failover_error(&Error::Timeout));
        assert!(is_failover_error(&Error::ModelNotReady("x".to_string())));
        assert!(is_failover_error(&Error::ResourceUnavailable(
            "x".to_string()
        )));
        assert!(!is_failover_error(&Error::Validation("x".to_string())));
        assert!(!is_failover_error(&Error::ModelNotFound("x".to_string())));
    }

    fn managed_model(id: &str, port: u16) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            backend: "llama.cpp".to_string(),
            capabilities: crate::models::capabilities::ModelCapabilities {
                supports_tools: false,
                supports_streaming: true,
                supports_structured_output: false,
                task_affinity: vec!["general".to_string()],
            },
            context_limit: 8192,
            memory_required_mb: 2048,
            quantization: Some("GGUF".to_string()),
            enabled: true,
            gguf_path: Some("E:/models/large.gguf".to_string()),
            local_port: Some(port),
            priority: 0,
        }
    }

    #[test]
    fn managed_port_defaults_to_base_url_port() {
        assert_eq!(default_llamacpp_port(&RuntimeConfig::default()), 8081);
        let custom = RuntimeConfig {
            llamacpp_base_url: "http://127.0.0.1:8099/".to_string(),
            ..RuntimeConfig::default()
        };
        assert_eq!(default_llamacpp_port(&custom), 8099);
        let unparseable = RuntimeConfig {
            llamacpp_base_url: "not-a-url".to_string(),
            ..RuntimeConfig::default()
        };
        assert_eq!(default_llamacpp_port(&unparseable), 8081);
    }

    #[tokio::test]
    async fn duplicate_managed_ports_skip_one_adapter() {
        let config = RuntimeConfig {
            max_concurrent_requests: 2,
            ..RuntimeConfig::default()
        };
        let models = vec![
            managed_model("a-local", 8091),
            managed_model("b-local", 8091),
        ];
        let runtime = Runtime::with_config(config, models).await.unwrap();
        // Install order follows the registry's hash order, so either model
        // may be the skipped one — but exactly one of them must be: the
        // skipped model fails with "no adapter", the installed one reaches
        // managed load and fails on the missing fixture .gguf instead.
        let mut outcomes = Vec::new();
        for id in ["a-local", "b-local"] {
            let mut req = request("hello");
            req.model = Some(id.to_string());
            outcomes.push(format!("{:?}", runtime.infer(req).await.unwrap_err()));
        }
        outcomes.sort();
        assert!(
            outcomes[0].contains("no adapter registered"),
            "outcomes: {outcomes:?}"
        );
        assert!(
            outcomes[1].contains("gguf file not found"),
            "outcomes: {outcomes:?}"
        );
    }

    #[tokio::test]
    async fn unknown_backend_installs_no_adapter() {
        let config = RuntimeConfig {
            max_concurrent_requests: 2,
            ..RuntimeConfig::default()
        };
        let models = vec![ModelInfo {
            id: "mystery".to_string(),
            name: "mystery".to_string(),
            backend: "does-not-exist".to_string(),
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
        }];
        let runtime = Runtime::with_config(config, models).await.unwrap();
        let mut req = request("hello");
        req.model = Some("mystery".to_string());
        let err = runtime.infer(req).await.unwrap_err();
        assert!(matches!(err, Error::Backend(_)), "got {err:?}");
    }
}
