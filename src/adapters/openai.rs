use crate::adapters::base::Adapter;
use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;

pub struct OpenAIAdapter {
    api_key: Option<String>,
    base_url: String,
}

impl OpenAIAdapter {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
}

#[async_trait]
impl Adapter for OpenAIAdapter {
    fn name(&self) -> &str {
        "openai"
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        // Mock implementation
        Ok(InferenceResponse {
            request_id: request.request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: request.model.clone().unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
            content: format!("OpenAI response: {}", request.prompt),
            finish_reason: Some("stop".to_string()),
            usage: Some(crate::core::types::Usage {
                prompt_tokens: request.prompt.len() as u32 / 4,
                completion_tokens: 50,
                total_tokens: request.prompt.len() as u32 / 4 + 50,
            }),
            tool_calls: None,
            metadata: None,
        })
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}
