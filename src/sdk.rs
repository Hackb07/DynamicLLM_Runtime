//! Stable Rust SDK for embedding the runtime in-process.
//!
//! [`Client`] owns a [`Runtime`](crate::core::runtime::Runtime) handle and
//! exposes the same operations as the HTTP API — inference, streaming
//! inference, agentic runs, model listing, and status — without going
//! through a socket. Use the builders ([`ClientBuilder`], [`InferBuilder`],
//! [`AgentTaskBuilder`]) to construct requests; they carry the documented
//! defaults so callers only set what they need.
//!
//! ```no_run
//! use dynamic_llm_runtime::sdk::{AgentTaskBuilder, Client, InferBuilder};
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let client = Client::builder().build().await?;
//! let response = client
//!     .infer(InferBuilder::new("Say hi").model("mock-gpt").build())
//!     .await?;
//! println!("{}", response.content);
//! let agent = client
//!     .run_agent(
//!         AgentTaskBuilder::new("Compute 2+3")
//!             .allow_tool("calc")
//!             .build(),
//!     )
//!     .await?;
//! println!("{}", agent.answer);
//! # Ok(())
//! # }
//! ```

use crate::adapters::ChunkStream;
use crate::agents::{AgentRequest, AgentResponse};
use crate::config::{ModelsFile, RuntimeConfig};
use crate::core::error::Result;
use crate::core::runtime::Runtime;
use crate::core::types::InferenceRequest;
use crate::core::types::InferenceResponse;
use crate::models::registry::ModelInfo;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// In-process runtime client.
///
/// Owns the [`Runtime`](crate::core::runtime::Runtime) behind an `Arc`,
/// so clones share one registry, scheduler, and telemetry store.
/// All methods mirror an HTTP endpoint (`/v1/infer`,
/// `/v1/infer/stream`, `/v1/agent/run`, `/v1/models`, `/v1/status`).
#[derive(Clone)]
pub struct Client {
    runtime: Arc<Runtime>,
}

impl Client {
    /// Build a client from explicit config and models.
    ///
    /// An empty `models` vec keeps the registry's built-in defaults
    /// (same rule as the server when `models.json` is absent).
    pub async fn new(config: RuntimeConfig, models: Vec<ModelInfo>) -> Result<Self> {
        Ok(Self {
            runtime: Arc::new(Runtime::with_config(config, models).await?),
        })
    }

    /// Load `runtime.json` / `models.json` from `dir` and build a client.
    /// Missing files fall back to the same defaults the server uses.
    pub async fn from_config_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = RuntimeConfig::load(dir);
        let models = match ModelsFile::load(dir) {
            Ok(Some(file)) => file.models,
            Ok(None) => Vec::new(),
            Err(e) => {
                tracing::warn!("{e}; using built-in model defaults");
                Vec::new()
            }
        };
        Self::new(config, models).await
    }

    /// Start a [`ClientBuilder`] with default config and no extra models.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Borrow the underlying runtime for advanced use (sessions,
    /// prefetch, workers) beyond the stable client surface.
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// Run one non-streaming inference request (mirrors `POST /v1/infer`).
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        self.runtime.infer(request).await
    }

    /// Run one streaming inference request (mirrors `POST /v1/infer/stream`).
    ///
    /// Uses the primary routing candidate only: failing over mid-stream
    /// would duplicate tokens.
    pub async fn infer_stream(&self, request: InferenceRequest) -> Result<ChunkStream> {
        self.runtime.infer_stream(request).await
    }

    /// Run one agentic task (mirrors `POST /v1/agent/run`).
    ///
    /// Terminal agent failures arrive as `Ok(response)` with `state` set
    /// and `error` populated (audit preserved); only malformed requests
    /// return `Err`.
    pub async fn run_agent(&self, request: AgentRequest) -> Result<AgentResponse> {
        self.runtime.run_agent(request).await
    }

    /// List registered models (mirrors `GET /v1/models`).
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.runtime.list_models().await
    }

    /// Full runtime status: loaded models, memory, scheduler, telemetry
    /// (mirrors `GET /v1/status`, the same payload the dashboard polls).
    pub async fn status(&self) -> serde_json::Value {
        self.runtime.get_status().await
    }
}

/// Builder for [`Client`]. Starts from [`RuntimeConfig::default`] with no
/// extra models; [`build`](ClientBuilder::build) is async because runtime
/// construction installs adapters under async locks.
#[derive(Debug, Default)]
pub struct ClientBuilder {
    config: RuntimeConfig,
    models: Vec<ModelInfo>,
}

impl ClientBuilder {
    /// Empty builder with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the whole runtime config.
    pub fn config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Replace the model list (empty = registry defaults).
    pub fn models(mut self, models: Vec<ModelInfo>) -> Self {
        self.models = models;
        self
    }

    /// Register one additional model.
    pub fn add_model(mut self, model: ModelInfo) -> Self {
        self.models.push(model);
        self
    }

    /// Bind host the server would listen on (kept for config parity;
    /// the in-process client itself never binds).
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    /// Bind port the server would listen on (config parity only).
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Per-request end-to-end timeout, in seconds.
    pub fn request_timeout_secs(mut self, secs: u64) -> Self {
        self.config.request_timeout_secs = secs;
        self
    }

    /// Maximum concurrent inference requests.
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    /// Routing strategy name (`rule_based`, `capability_based`,
    /// `resource_aware`, `fallback`, `adaptive`, `quality_aware`, `fixed`).
    pub fn routing_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.config.routing_strategy = strategy.into();
        self
    }

    /// Model id used when the routing strategy is `fixed`.
    pub fn fixed_model(mut self, model: impl Into<String>) -> Self {
        self.config.fixed_model = Some(model.into());
        self
    }

    /// Eviction policy name (`lru`, `lfu`, `cost_aware`, `priority_aware`).
    pub fn eviction_policy(mut self, policy: impl Into<String>) -> Self {
        self.config.eviction_policy = policy.into();
        self
    }

    /// Admission budget (MB) for resident local models.
    pub fn local_model_budget_mb(mut self, budget_mb: u64) -> Self {
        self.config.local_model_budget_mb = budget_mb;
        self
    }

    /// Retries per backend call for transient failures.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// How many routing candidates the runtime fails over across.
    pub fn max_fallback_models(mut self, max: usize) -> Self {
        self.config.max_fallback_models = max;
        self
    }

    /// Build the client.
    pub async fn build(self) -> Result<Client> {
        Client::new(self.config, self.models).await
    }
}

/// Builder for [`InferenceRequest`].
///
/// Only `prompt` is required; everything else defaults to `None`
/// (runtime defaults apply). [`build`](InferBuilder::build) never fails:
/// request validation happens in [`Client::infer`], which reports
/// problems as [`Error::Validation`](crate::core::error::Error::Validation).
#[derive(Debug, Default)]
pub struct InferBuilder {
    prompt: String,
    model: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    stream: Option<bool>,
    tools: Option<Vec<crate::core::types::Tool>>,
    request_id: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

impl InferBuilder {
    /// Start a request for `prompt`.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }

    /// Pin a model id (otherwise the router selects one).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Cap completion tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Nucleus sampling cutoff.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Stop sequences.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Hint that the route must support streaming.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Tool definitions the model may call. A non-empty list also routes
    /// to a tool-capable model; an empty list means no tool requirement.
    pub fn tools(mut self, tools: Vec<crate::core::types::Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Explicit request id for tracing (otherwise a UUID is assigned).
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Request metadata; `{"task": "<name>"}` routes by task affinity.
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add one metadata entry.
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Build the [`InferenceRequest`].
    pub fn build(self) -> InferenceRequest {
        InferenceRequest {
            request_id: self.request_id,
            prompt: self.prompt,
            model: self.model,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            stream: self.stream,
            tools: self.tools,
            metadata: self.metadata,
        }
    }
}

/// Builder for [`AgentRequest`].
///
/// Defaults mirror the HTTP API: `max_steps = 8`, `timeout_secs = 60`,
/// and a deny-by-default tool allowlist (empty = no tool may execute).
#[derive(Debug)]
pub struct AgentTaskBuilder {
    prompt: String,
    allowed_tools: Vec<String>,
    max_steps: u32,
    timeout_secs: u64,
    token_budget: Option<u32>,
    model: Option<String>,
    max_tokens_per_step: Option<u32>,
}

impl AgentTaskBuilder {
    /// Start a task with default limits and no allowed tools.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            allowed_tools: Vec::new(),
            max_steps: 8,
            timeout_secs: 60,
            token_budget: None,
            model: None,
            max_tokens_per_step: None,
        }
    }

    /// Allow one tool (may be called repeatedly; duplicates are kept as-is
    /// and matched by name at execution time).
    pub fn allow_tool(mut self, tool: impl Into<String>) -> Self {
        self.allowed_tools.push(tool.into());
        self
    }

    /// Replace the whole tool allowlist.
    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }

    /// Maximum agent steps (default 8).
    pub fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Per-run timeout in seconds (default 60).
    pub fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Total token budget across steps.
    pub fn token_budget(mut self, budget: u32) -> Self {
        self.token_budget = Some(budget);
        self
    }

    /// Model hint for the run's inference steps.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Per-step completion cap.
    pub fn max_tokens_per_step(mut self, max_tokens: u32) -> Self {
        self.max_tokens_per_step = Some(max_tokens);
        self
    }

    /// Build the [`AgentRequest`].
    pub fn build(self) -> AgentRequest {
        AgentRequest {
            prompt: self.prompt,
            allowed_tools: self.allowed_tools,
            max_steps: self.max_steps,
            timeout_secs: self.timeout_secs,
            token_budget: self.token_budget,
            model: self.model,
            max_tokens_per_step: self.max_tokens_per_step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_builder_defaults_to_prompt_only() {
        let request = InferBuilder::new("hello").build();
        assert_eq!(request.prompt, "hello");
        assert_eq!(request.model, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.temperature, None);
        assert_eq!(request.top_p, None);
        assert_eq!(request.stop, None);
        assert_eq!(request.stream, None);
        assert_eq!(request.metadata, None);
        assert_eq!(request.request_id, None);
    }

    #[test]
    fn infer_builder_sets_every_field() {
        let request = InferBuilder::new("hi")
            .model("m1")
            .max_tokens(32)
            .temperature(0.5)
            .top_p(0.9)
            .stop(vec!["END".to_string()])
            .stream(true)
            .tools(vec![crate::core::types::Tool {
                name: "calc".to_string(),
                description: "math".to_string(),
                parameters: serde_json::json!({}),
            }])
            .request_id("req-1")
            .metadata_entry("task", "summarization")
            .build();
        assert_eq!(request.model.as_deref(), Some("m1"));
        assert_eq!(request.max_tokens, Some(32));
        assert_eq!(request.temperature, Some(0.5));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.stop, Some(vec!["END".to_string()]));
        assert_eq!(request.stream, Some(true));
        assert_eq!(request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(request.tools.as_ref().unwrap()[0].name, "calc");
        assert_eq!(request.request_id.as_deref(), Some("req-1"));
        assert_eq!(
            request.metadata.as_ref().unwrap().get("task").unwrap(),
            "summarization"
        );
    }

    #[test]
    fn agent_builder_defaults_match_http_api() {
        let request = AgentTaskBuilder::new("do it").build();
        assert_eq!(request.prompt, "do it");
        assert!(request.allowed_tools.is_empty());
        assert_eq!(request.max_steps, 8);
        assert_eq!(request.timeout_secs, 60);
        assert_eq!(request.token_budget, None);
        assert_eq!(request.model, None);
        assert_eq!(request.max_tokens_per_step, None);
    }

    #[test]
    fn agent_builder_sets_limits_and_tools() {
        let request = AgentTaskBuilder::new("task")
            .allow_tool("calc")
            .allow_tool("echo")
            .max_steps(4)
            .timeout_secs(30)
            .token_budget(1000)
            .model("m1")
            .max_tokens_per_step(64)
            .build();
        assert_eq!(request.allowed_tools, vec!["calc", "echo"]);
        assert_eq!(request.max_steps, 4);
        assert_eq!(request.timeout_secs, 30);
        assert_eq!(request.token_budget, Some(1000));
        assert_eq!(request.model.as_deref(), Some("m1"));
        assert_eq!(request.max_tokens_per_step, Some(64));
    }

    #[test]
    fn agent_allowed_tools_replaces_list() {
        let request = AgentTaskBuilder::new("task")
            .allow_tool("calc")
            .allowed_tools(vec!["echo".to_string()])
            .build();
        assert_eq!(request.allowed_tools, vec!["echo"]);
    }

    #[tokio::test]
    async fn client_builder_builds_and_reports_status() {
        let client = Client::builder()
            .request_timeout_secs(5)
            .max_concurrent_requests(4)
            .routing_strategy("fixed")
            .fixed_model("mock-gpt")
            .eviction_policy("lfu")
            .local_model_budget_mb(1024)
            .max_retries(1)
            .max_fallback_models(1)
            .build()
            .await
            .unwrap();
        assert_eq!(client.runtime().config().request_timeout_secs, 5);
        assert_eq!(client.runtime().config().max_concurrent_requests, 4);
        assert_eq!(client.runtime().config().routing_strategy, "fixed");
        assert_eq!(
            client.runtime().config().fixed_model.as_deref(),
            Some("mock-gpt")
        );
        assert_eq!(client.runtime().config().eviction_policy, "lfu");
        assert_eq!(client.runtime().config().local_model_budget_mb, 1024);
        assert_eq!(client.runtime().config().max_retries, 1);
        assert_eq!(client.runtime().config().max_fallback_models, 1);
        let status = client.status().await;
        assert_eq!(status["status"], "running");
        assert!(status.get("telemetry").is_some());
        assert!(!client.list_models().await.is_empty());
    }

    #[tokio::test]
    async fn client_rejects_empty_prompt() {
        let client = Client::builder().build().await.unwrap();
        let error = client
            .infer(InferBuilder::new("   ").build())
            .await
            .unwrap_err();
        assert!(matches!(error, crate::core::error::Error::Validation(_)));
    }
}
