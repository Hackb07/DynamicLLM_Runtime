use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;

#[async_trait]
pub trait Adapter: Send + Sync {
    fn name(&self) -> &str;
    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse>;
    async fn health_check(&self) -> Result<bool>;
}
