use crate::core::error::{Error, Result};
use crate::models::registry::ModelInfo;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Runtime-wide settings. Loaded from `config/runtime.json` with
/// environment-variable overrides applied on top (see [`RuntimeConfig::load`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub host: String,
    pub port: u16,
    /// Per-request end-to-end timeout (seconds), applied around backend calls.
    pub request_timeout_secs: u64,
    /// Retries per backend call for transient failures (429/5xx/connect errors).
    pub max_retries: u32,
    /// Base backoff between retries (milliseconds); doubled each attempt.
    pub retry_backoff_ms: u64,
    /// Maximum concurrent inference requests; excess waits briefly, then fails.
    pub max_concurrent_requests: usize,
    /// How many routing candidates the runtime will fail over across.
    pub max_fallback_models: usize,
    /// Admission budget (MB) for resident local models; see `ModelManager`.
    pub local_model_budget_mb: u64,
    /// One of: fixed, rule_based, capability_based, resource_aware,
    /// fallback, adaptive, quality_aware.
    pub routing_strategy: String,
    /// One of: lru, lfu, cost_aware, priority_aware.
    pub eviction_policy: String,
    /// Explicit model id used when `routing_strategy` is "fixed".
    #[serde(default)]
    pub fixed_model: Option<String>,
    pub openai_base_url: String,
    pub openai_default_model: String,
    /// API key; prefer the `OPENAI_API_KEY` env var over storing it here.
    #[serde(default)]
    pub openai_api_key: Option<String>,
    pub llamacpp_base_url: String,
    pub llamacpp_default_model: String,
    /// `llama-server` executable used for managed local models (resolved via
    /// `PATH` when relative). Only used when a model sets `gguf_path`.
    #[serde(default = "default_llamacpp_binary")]
    pub llamacpp_binary: String,
    /// Context size passed to managed servers (`--ctx-size`). The KV cache
    /// grows with this value, so the default is a modest 4096 for min-RAM
    /// operation; raise it per workload needs.
    #[serde(default = "default_llamacpp_ctx_size")]
    pub llamacpp_ctx_size: u32,
    /// Extra argv entries appended to the managed `llama-server` command
    /// (e.g. `["--threads", "8"]`). Managed servers always pass
    /// `--mmap` and never `--mlock` first, so weights stay SSD-backed.
    #[serde(default)]
    pub llamacpp_extra_args: Vec<String>,
    /// GPU layers offloaded by llama.cpp (`--n-gpu-layers` for managed
    /// servers, `n_gpu_layers` for the in-process engine). `0` (default)
    /// is CPU-only and always safe; set higher only with a CUDA-capable
    /// build and GPU. Overridable with `LLAMACPP_N_GPU_LAYERS`.
    #[serde(default)]
    pub n_gpu_layers: u32,
    /// Weight of the latency term in adaptive/quality-aware routing.
    /// Higher prefers faster models more strongly. Old configs omit
    /// this and get the default via serde.
    #[serde(default = "default_routing_latency_weight")]
    pub routing_latency_weight: f64,
    /// Weight of the success-rate term in adaptive/quality-aware routing.
    #[serde(default = "default_routing_success_weight")]
    pub routing_success_weight: f64,
    /// Bonus added when a candidate matches the request task affinity.
    #[serde(default = "default_routing_affinity_bonus")]
    pub routing_affinity_bonus: f64,
    /// Max concurrent background prefetches (bounded to avoid load storms).
    #[serde(default = "default_prefetch_concurrency")]
    pub prefetch_concurrency: usize,
    /// Model ids to prefetch in the background at startup (warm set).
    #[serde(default)]
    pub prefetch_warm_models: Vec<String>,
    /// When true, the frequency predictor prefetches its next-model guess
    /// after each request. Defaults to false to preserve prior behavior.
    #[serde(default)]
    pub predictive_prefetch_enabled: bool,
    /// How many recent model ids the predictor retains (bounded memory).
    #[serde(default = "default_predictor_history_limit")]
    pub predictor_history_limit: usize,
    /// Remote inference workers (opt-in remote execution). Empty by
    /// default: with no workers configured the runtime behaves
    /// single-node, exactly as before. Each entry: id, addr (base URL),
    /// models it serves (empty = any), and concurrent-request capacity.
    #[serde(default)]
    pub workers: Vec<WorkerConfig>,
}

/// One remote worker entry in `RuntimeConfig::workers`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub id: String,
    pub addr: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_worker_capacity")]
    pub capacity: usize,
}

fn default_worker_capacity() -> usize {
    4
}

fn default_llamacpp_binary() -> String {
    "llama-server".to_string()
}

fn default_llamacpp_ctx_size() -> u32 {
    4096
}

fn default_routing_latency_weight() -> f64 {
    1.0
}

fn default_routing_success_weight() -> f64 {
    1.0
}

fn default_routing_affinity_bonus() -> f64 {
    0.5
}

fn default_prefetch_concurrency() -> usize {
    2
}

fn default_predictor_history_limit() -> usize {
    100
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            request_timeout_secs: 120,
            max_retries: 3,
            retry_backoff_ms: 200,
            max_concurrent_requests: 16,
            max_fallback_models: 2,
            local_model_budget_mb: 8192,
            routing_strategy: "rule_based".to_string(),
            eviction_policy: "lru".to_string(),
            fixed_model: None,
            openai_base_url: "https://api.openai.com/v1".to_string(),
            openai_default_model: "gpt-3.5-turbo".to_string(),
            openai_api_key: None,
            llamacpp_base_url: "http://127.0.0.1:8081".to_string(),
            llamacpp_default_model: "llama-3.2-3b".to_string(),
            llamacpp_binary: default_llamacpp_binary(),
            llamacpp_ctx_size: default_llamacpp_ctx_size(),
            llamacpp_extra_args: Vec::new(),
            n_gpu_layers: 0,
            routing_latency_weight: default_routing_latency_weight(),
            routing_success_weight: default_routing_success_weight(),
            routing_affinity_bonus: default_routing_affinity_bonus(),
            prefetch_concurrency: default_prefetch_concurrency(),
            prefetch_warm_models: Vec::new(),
            predictive_prefetch_enabled: false,
            predictor_history_limit: default_predictor_history_limit(),
            workers: Vec::new(),
        }
    }
}

impl RuntimeConfig {
    /// Directory holding `runtime.json` / `models.json`.
    /// Overridable with `DLLM_CONFIG_DIR`.
    pub fn config_dir() -> String {
        std::env::var("DLLM_CONFIG_DIR").unwrap_or_else(|_| "config".to_string())
    }

    /// Load from `dir/runtime.json` (or defaults when the file is missing),
    /// then apply environment overrides.
    pub fn load(dir: impl AsRef<Path>) -> Self {
        let path = dir.as_ref().join("runtime.json");
        let mut config = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str(&text) {
                Ok(config) => config,
                Err(e) => {
                    tracing::warn!(
                        "Invalid {} ({}); using default runtime config",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!("No {} found; using default runtime config", path.display());
                Self::default()
            }
            Err(e) => {
                tracing::warn!(
                    "Cannot read {} ({}); using default runtime config",
                    path.display(),
                    e
                );
                Self::default()
            }
        };
        config.apply_env_overrides();
        if let Err(e) = config.validate() {
            tracing::warn!("Invalid runtime config ({e}); using defaults");
            return Self::default();
        }
        config
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(host) = std::env::var("DLLM_HOST") {
            self.host = host;
        }
        if let Ok(port) = std::env::var("DLLM_PORT") {
            if let Ok(port) = port.parse() {
                self.port = port;
            }
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            if !key.trim().is_empty() {
                self.openai_api_key = Some(key);
            }
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            if !url.trim().is_empty() {
                self.openai_base_url = url;
            }
        }
        if let Ok(url) = std::env::var("LLAMACPP_BASE_URL") {
            if !url.trim().is_empty() {
                self.llamacpp_base_url = url;
            }
        }
        if let Ok(binary) = std::env::var("LLAMACPP_BINARY") {
            if !binary.trim().is_empty() {
                self.llamacpp_binary = binary;
            }
        }
        if let Ok(ctx) = std::env::var("LLAMACPP_CTX_SIZE") {
            if let Ok(ctx) = ctx.parse() {
                self.llamacpp_ctx_size = ctx;
            }
        }
        if let Ok(layers) = std::env::var("LLAMACPP_N_GPU_LAYERS") {
            if let Ok(layers) = layers.parse() {
                self.n_gpu_layers = layers;
            }
        }
        if let Ok(secs) = std::env::var("DLLM_REQUEST_TIMEOUT_SECS") {
            if let Ok(secs) = secs.parse() {
                self.request_timeout_secs = secs;
            }
        }
        if let Ok(retries) = std::env::var("DLLM_MAX_RETRIES") {
            if let Ok(retries) = retries.parse() {
                self.max_retries = retries;
            }
        }
    }

    fn validate(&self) -> Result<()> {
        if self.port == 0 {
            return Err(Error::Config("port must be non-zero".to_string()));
        }
        if self.request_timeout_secs == 0 {
            return Err(Error::Config(
                "request_timeout_secs must be non-zero".to_string(),
            ));
        }
        if self.max_concurrent_requests == 0 {
            return Err(Error::Config(
                "max_concurrent_requests must be non-zero".to_string(),
            ));
        }
        if self.max_fallback_models == 0 {
            return Err(Error::Config(
                "max_fallback_models must be non-zero".to_string(),
            ));
        }
        for (name, weight) in [
            ("routing_latency_weight", self.routing_latency_weight),
            ("routing_success_weight", self.routing_success_weight),
            ("routing_affinity_bonus", self.routing_affinity_bonus),
        ] {
            if !weight.is_finite() || weight < 0.0 {
                return Err(Error::Config(format!("{name} must be finite and >= 0")));
            }
        }
        if self.prefetch_concurrency == 0 {
            return Err(Error::Config(
                "prefetch_concurrency must be non-zero".to_string(),
            ));
        }
        if self.predictor_history_limit == 0 {
            return Err(Error::Config(
                "predictor_history_limit must be non-zero".to_string(),
            ));
        }
        for worker in &self.workers {
            if worker.id.trim().is_empty() {
                return Err(Error::Config("worker id must not be empty".to_string()));
            }
            if worker.addr.trim().is_empty() {
                return Err(Error::Config(format!(
                    "worker '{}' addr must not be empty",
                    worker.id
                )));
            }
            if worker.capacity == 0 {
                return Err(Error::Config(format!(
                    "worker '{}' capacity must be non-zero",
                    worker.id
                )));
            }
        }
        Ok(())
    }

    /// Resolve the OpenAI API key from config or environment.
    pub fn openai_api_key(&self) -> Option<String> {
        self.openai_api_key.clone().or_else(|| {
            std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|key| !key.trim().is_empty())
        })
    }
}

/// Contents of `config/models.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsFile {
    pub models: Vec<ModelInfo>,
}

impl ModelsFile {
    /// Load from `dir/models.json`. Returns `Ok(None)` when the file is
    /// missing so callers can fall back to built-in defaults.
    pub fn load(dir: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = dir.as_ref().join("models.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let file: Self = serde_json::from_str(&text)
                    .map_err(|e| Error::Config(format!("invalid {}: {e}", path.display())))?;
                Ok(Some(file))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Config(format!(
                "cannot read {}: {e}",
                path.display()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        RuntimeConfig::default().validate().unwrap();
    }

    #[test]
    fn invalid_values_are_rejected() {
        let config = RuntimeConfig {
            port: 0,
            ..RuntimeConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
