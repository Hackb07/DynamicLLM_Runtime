use crate::adapters::base::{Adapter, ChunkStream};
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse, Usage};
use crate::engine::{EngineConfig, GenerateOptions, LocalEngine};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

/// In-process inference backend (`ModelInfo.backend == "local"`).
///
/// Loads the model's `.gguf` weights into this process (SSD-backed `mmap`)
/// and generates tokens locally. All llama.cpp calls run on Tokio's
/// blocking pool; the async surface here never blocks the executor.
pub struct LocalEngineAdapter {
    config: EngineConfig,
    default_model: String,
    engine: Mutex<Option<Arc<LocalEngine>>>,
}

impl LocalEngineAdapter {
    pub fn new(gguf_path: &str, default_model: &str, n_ctx: u32) -> Self {
        Self {
            config: EngineConfig::new(gguf_path, n_ctx),
            default_model: default_model.to_string(),
            engine: Mutex::new(None),
        }
    }

    /// Offload `n` transformer layers to GPU (default 0 = CPU-only).
    pub fn with_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.config.n_gpu_layers = n_gpu_layers;
        self
    }

    async fn loaded(&self) -> Result<Arc<LocalEngine>> {
        self.engine
            .lock()
            .await
            .clone()
            .ok_or_else(|| Error::ModelNotReady(format!("{} is not loaded", self.default_model)))
    }

    fn options_for(&self, request: &InferenceRequest) -> GenerateOptions {
        let mut opts = GenerateOptions::new(request.prompt.clone());
        if let Some(max_tokens) = request.max_tokens {
            opts.max_tokens = max_tokens.max(1);
        }
        if let Some(temperature) = request.temperature {
            opts.temperature = temperature;
        }
        if let Some(top_p) = request.top_p {
            opts.top_p = top_p;
        }
        opts
    }
}

#[async_trait]
impl Adapter for LocalEngineAdapter {
    fn name(&self) -> &str {
        "local-engine"
    }

    fn backend(&self) -> &str {
        "local"
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let engine = self.loaded().await?;
        let opts = self.options_for(request);
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.default_model.clone());
        let generated = tokio::task::spawn_blocking(move || engine.generate(&opts))
            .await
            .map_err(|e| Error::Backend(format!("inference task failed: {e}")))??;
        Ok(InferenceResponse {
            request_id: request
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model,
            content: generated.text,
            finish_reason: Some(
                if generated.stopped_at_eog {
                    "stop"
                } else {
                    "length"
                }
                .to_string(),
            ),
            usage: Some(Usage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated
                    .prompt_tokens
                    .saturating_add(generated.completion_tokens),
            }),
            tool_calls: None,
            metadata: None,
        })
    }

    async fn infer_stream(&self, request: &InferenceRequest) -> Result<ChunkStream> {
        let engine = self.loaded().await?;
        let opts = self.options_for(request);
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(32);
        tokio::task::spawn_blocking(move || {
            let result = engine.generate_stream(&opts, |piece| {
                let _ = tx.blocking_send(Ok(piece.to_string()));
            });
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });
        let stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(stream))
    }

    async fn load(&self) -> Result<()> {
        if self.engine.lock().await.is_some() {
            return Ok(());
        }
        let config = self.config.clone();
        let engine = tokio::task::spawn_blocking(move || LocalEngine::load(config))
            .await
            .map_err(|e| Error::Backend(format!("model load task failed: {e}")))??;
        *self.engine.lock().await = Some(Arc::new(engine));
        Ok(())
    }

    async fn unload(&self) -> Result<()> {
        *self.engine.lock().await = None;
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.engine.lock().await.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> LocalEngineAdapter {
        LocalEngineAdapter::new("E:/models/does-not-exist.gguf", "local-test", 512)
    }

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
    fn request_options_map_onto_generation() {
        let adapter = adapter();
        let mut req = request("hi");
        req.max_tokens = Some(64);
        req.temperature = Some(0.5);
        req.top_p = Some(0.7);
        let opts = adapter.options_for(&req);
        assert_eq!(opts.prompt, "hi");
        assert_eq!(opts.max_tokens, 64);
        assert_eq!(opts.temperature, 0.5);
        assert_eq!(opts.top_p, 0.7);

        let defaults = adapter.options_for(&request("hi"));
        assert!(defaults.max_tokens >= 1);
    }

    #[tokio::test]
    async fn load_reports_a_missing_gguf_file() {
        let adapter = adapter();
        let err = adapter.load().await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)), "got {err:?}");
        assert!(!adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn infer_before_load_is_not_ready() {
        let adapter = adapter();
        let err = adapter.infer(&request("hi")).await.unwrap_err();
        assert!(matches!(err, Error::ModelNotReady(_)), "got {err:?}");
        let err = adapter
            .infer_stream(&request("hi"))
            .await
            .err()
            .expect("expected ModelNotReady");
        assert!(matches!(err, Error::ModelNotReady(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn unload_without_load_is_a_noop() {
        let adapter = adapter();
        adapter.unload().await.unwrap();
        assert!(!adapter.health_check().await.unwrap());
    }
}
