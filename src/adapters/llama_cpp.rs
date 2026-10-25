use crate::adapters::base::Adapter;
use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;

pub struct LlamaCppAdapter {
    model_path: Option<String>,
}

impl LlamaCppAdapter {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }
}

#[async_trait]
impl Adapter for LlamaCppAdapter {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        Ok(InferenceResponse {
            request_id: request.request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: request.model.clone().unwrap_or_else(|| "llama-3.2-3b".to_string()),
            content: format!("LLaMA.cpp response: {}", request.prompt),
            finish_reason: Some("stop".to_string()),
            usage: Some(crate::core::types::Usage {
                prompt_tokens: request.prompt.len() as u32 / 4,
                completion_tokens: 40,
                total_tokens: request.prompt.len() as u32 / 4 + 40,
            }),
            tool_calls: None,
            metadata: None,
        })
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.model_path.is_some())
    }
}
