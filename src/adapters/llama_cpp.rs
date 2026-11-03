use crate::adapters::base::{Adapter, ChunkStream};
use crate::adapters::http::{to_inference_response, OpenAiCompatClient};
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;
use std::time::{Duration, Instant};

/// How to spawn a runtime-managed `llama-server` for one local model.
#[derive(Debug, Clone)]
pub struct ManagedServerConfig {
    /// `llama-server` executable (resolved via `PATH` when relative).
    pub binary: String,
    /// Path to the `.gguf` weight file passed as `-m`.
    pub gguf_path: String,
    /// Port passed as `--port`; the HTTP client targets it.
    pub port: u16,
    /// Context size passed as `--ctx-size`.
    pub ctx_size: u32,
    /// GPU layers passed as `--n-gpu-layers` (only when > 0; 0 = CPU-only).
    pub n_gpu_layers: u32,
    /// Extra argv entries appended after the built-in flags.
    pub extra_args: Vec<String>,
}

/// Build the `llama-server` argv for SSD-backed, minimum-RAM operation.
///
/// Weights stay memory-mapped from disk (`--mmap`, never `--mlock`), so the
/// OS pages them from SSD on demand instead of pinning the whole model in
/// RAM. This lets a model larger than available RAM run, at the cost of
/// slower tokens when pages must be faulted in.
pub fn server_args(config: &ManagedServerConfig) -> Vec<String> {
    let mut args = vec![
        "-m".to_string(),
        config.gguf_path.clone(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        config.port.to_string(),
        "--ctx-size".to_string(),
        config.ctx_size.to_string(),
        "--mmap".to_string(),
    ];
    if config.n_gpu_layers > 0 {
        args.push("--n-gpu-layers".to_string());
        args.push(config.n_gpu_layers.to_string());
    }
    args.extend(config.extra_args.iter().cloned());
    args
}

/// Adapter for a locally-hosted llama.cpp server (`llama-server`), using its
/// OpenAI-compatible `/v1/chat/completions` endpoint and `/health` probe.
///
/// Two modes:
/// - Unmanaged (`from_config`): talk to an externally started server, e.g.
///   `llama-server -m model.gguf --port 8081`. `load` only health-checks.
/// - Managed (`managed`): spawn `llama-server -m <gguf>` on `load` with
///   SSD-backed `--mmap` flags and stop it on `unload`.
pub struct LlamaCppAdapter {
    client: OpenAiCompatClient,
    default_model: String,
    managed: Option<ManagedServerConfig>,
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    startup_timeout: Duration,
}

impl LlamaCppAdapter {
    pub fn new(base_url: Option<String>, model: Option<String>) -> Result<Self> {
        Self::from_config(
            base_url.as_deref().unwrap_or("http://127.0.0.1:8081"),
            model.as_deref().unwrap_or("llama-3.2-3b"),
            Duration::from_secs(300),
            1,
            200,
        )
    }

    pub fn from_config(
        base_url: &str,
        default_model: &str,
        timeout: Duration,
        max_retries: u32,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let client = OpenAiCompatClient::new(
            "llama.cpp",
            base_url,
            "/v1/chat/completions",
            "/health",
            None,
            timeout,
            max_retries,
            backoff_base_ms,
        )?;
        Ok(Self {
            client,
            default_model: default_model.to_string(),
            managed: None,
            child: tokio::sync::Mutex::new(None),
            startup_timeout: Duration::from_secs(300),
        })
    }

    /// Build an adapter that spawns and owns its own `llama-server`.
    /// The HTTP client targets `http://127.0.0.1:{port}`.
    pub fn managed(
        managed: ManagedServerConfig,
        default_model: &str,
        timeout: Duration,
        max_retries: u32,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let base_url = format!("http://127.0.0.1:{}", managed.port);
        let mut adapter = Self::from_config(
            &base_url,
            default_model,
            timeout,
            max_retries,
            backoff_base_ms,
        )?;
        adapter.managed = Some(managed);
        Ok(adapter)
    }

    /// Override how long `load` waits for a spawned server to turn healthy.
    /// Large models loading from SSD can take minutes; tests use a short
    /// deadline instead.
    pub fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    pub fn is_managed(&self) -> bool {
        self.managed.is_some()
    }

    /// Full command line (`[binary, args...]`) this adapter would spawn,
    /// or `None` for unmanaged adapters.
    pub fn server_command_line(&self) -> Option<Vec<String>> {
        self.managed.as_ref().map(|config| {
            let mut command = vec![config.binary.clone()];
            command.extend(server_args(config));
            command
        })
    }

    fn model_for(&self, request: &InferenceRequest) -> String {
        request
            .model
            .clone()
            .unwrap_or_else(|| self.default_model.clone())
    }

    /// Spawn the managed server when needed, then wait for `/health`.
    async fn load_managed(&self, config: &ManagedServerConfig) -> Result<()> {
        if !std::path::Path::new(&config.gguf_path).is_file() {
            return Err(Error::ResourceUnavailable(format!(
                "gguf file not found: {}",
                config.gguf_path
            )));
        }
        // Another process (or a previous load) may already serve this port.
        if self.client.health().await {
            tracing::info!(
                "llama-server already healthy on port {}; reusing it",
                config.port
            );
            return Ok(());
        }
        {
            let mut guard = self.child.lock().await;
            let running = match guard.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(None) => true,
                    // Previously spawned process exited; reap it and respawn.
                    _ => {
                        guard.take();
                        false
                    }
                },
                None => false,
            };
            if !running {
                let args = server_args(config);
                let mut command = tokio::process::Command::new(&config.binary);
                command
                    .args(&args)
                    .kill_on_drop(true)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::inherit());
                let child = command.spawn().map_err(|e| {
                    Error::ResourceUnavailable(format!(
                        "cannot start '{}': {e}; install llama.cpp or set llamacpp_binary",
                        config.binary
                    ))
                })?;
                tracing::info!(
                    "started managed llama-server: {} {}",
                    config.binary,
                    args.join(" ")
                );
                *guard = Some(child);
            }
        }
        self.wait_for_healthy(config, Instant::now() + self.startup_timeout)
            .await
    }

    /// Poll `/health` until the deadline, surfacing an early server exit
    /// (bad flags, port in use, unreadable weights) instead of timing out.
    async fn wait_for_healthy(
        &self,
        config: &ManagedServerConfig,
        deadline: Instant,
    ) -> Result<()> {
        loop {
            if self.client.health().await {
                return Ok(());
            }
            let exited = {
                let mut guard = self.child.lock().await;
                match guard.as_mut() {
                    Some(child) => match child.try_wait() {
                        Ok(status) => status,
                        Err(e) => {
                            return Err(Error::Backend(format!(
                                "managed llama-server wait failed: {e}"
                            )))
                        }
                    },
                    None => {
                        return Err(Error::Backend(
                            "managed llama-server handle lost".to_string(),
                        ))
                    }
                }
            };
            if let Some(status) = exited {
                self.child.lock().await.take();
                return Err(Error::ResourceUnavailable(format!(
                    "managed llama-server exited during startup (status: {status}); \
                     check the .gguf path and that port {} is free",
                    config.port
                )));
            }
            if Instant::now() >= deadline {
                return Err(Error::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}

#[async_trait]
impl Adapter for LlamaCppAdapter {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn backend(&self) -> &str {
        "llama.cpp"
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let model = self.model_for(request);
        let response = self.client.chat(&model, request).await?;
        Ok(to_inference_response(request, &model, response))
    }

    async fn infer_stream(&self, request: &InferenceRequest) -> Result<ChunkStream> {
        let model = self.model_for(request);
        self.client.chat_stream(&model, request).await
    }

    async fn load(&self) -> Result<()> {
        if let Some(config) = self.managed.clone() {
            return self.load_managed(&config).await;
        }
        if self.client.health().await {
            Ok(())
        } else {
            Err(Error::ResourceUnavailable(
                "llama.cpp server is not reachable; start `llama-server` first or set gguf_path for managed mode".to_string(),
            ))
        }
    }

    async fn unload(&self) -> Result<()> {
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            tracing::info!("stopping managed llama-server");
            child
                .kill()
                .await
                .map_err(|e| Error::Backend(format!("failed to stop managed llama-server: {e}")))?;
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.client.health().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_managed(gguf_path: &str, port: u16) -> ManagedServerConfig {
        ManagedServerConfig {
            binary: "llama-server".to_string(),
            gguf_path: gguf_path.to_string(),
            port,
            ctx_size: 4096,
            n_gpu_layers: 0,
            extra_args: Vec::new(),
        }
    }

    #[test]
    fn server_args_use_ssd_backed_mmap_without_mlock() {
        let mut config = test_managed("E:/models/large.gguf", 8081);
        config.extra_args = vec!["--n-gpu-layers".to_string(), "0".to_string()];
        let args = server_args(&config);
        assert_eq!(
            &args[..9],
            &[
                "-m",
                "E:/models/large.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
                "--ctx-size",
                "4096",
                "--mmap",
            ]
        );
        assert!(!args
            .iter()
            .any(|arg| arg == "--mlock" || arg == "--no-mmap"));
        assert_eq!(&args[9..], &["--n-gpu-layers", "0"]);
    }

    #[test]
    fn server_args_add_gpu_flag_only_when_positive() {
        let cpu = test_managed("E:/models/m.gguf", 8081);
        let args = server_args(&cpu);
        assert!(!args.iter().any(|arg| arg == "--n-gpu-layers"));

        let mut gpu = test_managed("E:/models/m.gguf", 8081);
        gpu.n_gpu_layers = 35;
        let args = server_args(&gpu);
        let pos = args.iter().position(|arg| arg == "--n-gpu-layers").unwrap();
        assert_eq!(args[pos + 1], "35");
    }

    #[tokio::test]
    async fn managed_load_rejects_missing_gguf_file() {
        let adapter = LlamaCppAdapter::managed(
            test_managed("E:/models/does-not-exist.gguf", 18081),
            "managed-test",
            Duration::from_secs(5),
            0,
            5,
        )
        .unwrap()
        .with_startup_timeout(Duration::from_secs(2));
        assert!(adapter.is_managed());
        let err = adapter.load().await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)), "got {err:?}");
        assert!(err.to_string().contains("does-not-exist.gguf"));
        // Nothing was spawned, so unload is a no-op.
        adapter.unload().await.unwrap();
    }

    #[tokio::test]
    async fn managed_load_reports_unusable_binary() {
        // A real (empty) file passes the existence check; the missing binary
        // then fails the spawn immediately without waiting for health.
        let gguf = tempfile::NamedTempFile::new().unwrap();
        let config = ManagedServerConfig {
            binary: "dllm-definitely-not-a-binary".to_string(),
            ..test_managed(gguf.path().to_str().unwrap(), 18082)
        };
        let adapter =
            LlamaCppAdapter::managed(config, "managed-test", Duration::from_secs(5), 0, 5)
                .unwrap()
                .with_startup_timeout(Duration::from_secs(2));
        let err = adapter.load().await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)), "got {err:?}");
        assert!(err.to_string().contains("dllm-definitely-not-a-binary"));
    }

    #[test]
    fn unmanaged_adapter_has_no_command_line() {
        let adapter = LlamaCppAdapter::new(None, None).unwrap();
        assert!(!adapter.is_managed());
        assert!(adapter.server_command_line().is_none());
    }
}
