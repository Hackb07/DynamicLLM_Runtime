use crate::adapters::Adapter;
use crate::core::error::{Error, Result};
use crate::core::types::{InferenceRequest, InferenceResponse};
use crate::models::lifecycle::ModelState;
use crate::models::registry::ModelRegistry;
use crate::telemetry::Telemetry;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelManager {
    registry: Arc<RwLock<ModelRegistry>>,
    telemetry: Arc<Telemetry>,
    states: Arc<RwLock<HashMap<String, ModelState>>>,
    adapters: Arc<RwLock<HashMap<String, Box<dyn Adapter>>>>,
}

impl ModelManager {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>, telemetry: Arc<Telemetry>) -> Self {
        Self {
            registry,
            telemetry,
            states: Arc::new(RwLock::new(HashMap::new())),
            adapters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn ensure_loaded(&self, model_id: &str) -> Result<()> {
        let mut states = self.states.write().await;
        let state = states.entry(model_id.to_string()).or_insert(ModelState::Unloaded);
        
        if *state == ModelState::Ready {
            return Ok(());
        }
        
        if *state == ModelState::Loading {
            // In a real implementation, wait for loading to complete
            return Ok(());
        }
        
        *state = ModelState::Loading;
        self.telemetry.record_model_load_start(model_id).await;
        
        // Simulate loading
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        *state = ModelState::Ready;
        self.telemetry.record_model_load_end(model_id).await;
        
        Ok(())
    }

    pub async fn infer(&self, model_id: &str, request: &InferenceRequest) -> Result<InferenceResponse> {
        let states = self.states.read().await;
        if states.get(model_id) != Some(&ModelState::Ready) {
            return Err(Error::ModelNotReady(model_id.to_string()));
        }
        drop(states);
        
        // Use adapter if available, otherwise return mock response
        let adapters = self.adapters.read().await;
        if let Some(adapter) = adapters.get(model_id) {
            return adapter.infer(request).await;
        }
        drop(adapters);
        
        // Mock response
        Ok(InferenceResponse {
            request_id: request.request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: model_id.to_string(),
            content: format!("Mock response from {} for: {}", model_id, request.prompt.chars().take(50).collect::<String>()),
            finish_reason: Some("stop".to_string()),
            usage: Some(crate::core::types::Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
            tool_calls: None,
            metadata: None,
        })
    }

    pub async fn unload(&self, model_id: &str) -> Result<()> {
        let mut states = self.states.write().await;
        states.insert(model_id.to_string(), ModelState::Unloaded);
        Ok(())
    }
}
