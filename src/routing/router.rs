use crate::core::error::{Error, Result};
use crate::core::types::InferenceRequest;
use crate::models::registry::ModelRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelRouter {
    registry: Arc<RwLock<ModelRegistry>>,
}

impl ModelRouter {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>) -> Self {
        Self { registry }
    }

    pub async fn route(&self, request: &InferenceRequest) -> Result<String> {
        // If model specified in request, use it
        if let Some(model) = &request.model {
            let registry = self.registry.read().await;
            registry.get(model)?;
            return Ok(model.clone());
        }

        // Default routing: prefer local if available, else cloud
        let registry = self.registry.read().await;
        let models = registry.list_models();
        
        // Try to find a local model first
        for model in &models {
            if model.backend == "llama.cpp" && model.enabled {
                return Ok(model.id.clone());
            }
        }
        
        // Fall back to any enabled model
        for model in &models {
            if model.enabled {
                return Ok(model.id.clone());
            }
        }
        
        Err(Error::Routing("No available models".to_string()))
    }
}
