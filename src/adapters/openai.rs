use crate::adapters::base::{Adapter, ChunkStream};
use crate::adapters::http::{to_inference_response, OpenAiCompatClient};
use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;
use std::time::Duration;

/// Adapter for the OpenAI chat-completions API (and compatible endpoints).
pub struct OpenAIAdapter {
    client: OpenAiCompatClient,
    default_model: String,
}

impl OpenAIAdapter {
    /// Build with explicit settings. `api_key` falls back to `OPENAI_API_KEY`.
    pub fn new(api_key: Option<String>, base_url: Option<String>) -> Result<Self> {
        Self::from_config(
            base_url.as_deref().unwrap_or("https://api.openai.com/v1"),
            api_key.or_else(|| {
                std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|key| !key.trim().is_empty())
            }),
            "gpt-3.5-turbo",
            Duration::from_secs(120),
            3,
            200,
        )
    }

    pub fn from_config(
        base_url: &str,
        api_key: Option<String>,
        default_model: &str,
        timeout: Duration,
        max_retries: u32,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let client = OpenAiCompatClient::new(
            "openai",
            base_url,
            "/chat/completions",
            "/models",
            api_key,
            timeout,
            max_retries,
            backoff_base_ms,
        )?;
        Ok(Self {
            client,
            default_model: default_model.to_string(),
        })
    }

    fn model_for(&self, request: &InferenceRequest) -> String {
        request
            .model
            .clone()
            .unwrap_or_else(|| self.default_model.clone())
    }
}

#[async_trait]
impl Adapter for OpenAIAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    fn backend(&self) -> &str {
        "openai"
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

    async fn health_check(&self) -> Result<bool> {
        Ok(self.client.health().await)
    }
}
