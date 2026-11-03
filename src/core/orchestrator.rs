use crate::core::error::Result;
use crate::core::runtime::Runtime;
use crate::core::types::{InferenceRequest, InferenceResponse};
use std::sync::Arc;

pub struct Orchestrator {
    runtime: Arc<Runtime>,
}

impl Orchestrator {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    pub async fn execute(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        self.runtime.infer(request).await
    }
}
