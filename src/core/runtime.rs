use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use crate::models::registry::ModelRegistry;
use crate::models::manager::ModelManager;
use crate::routing::router::ModelRouter;
use crate::telemetry::Telemetry;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct Runtime {
    registry: Arc<RwLock<ModelRegistry>>,
    manager: Arc<ModelManager>,
    router: Arc<ModelRouter>,
    telemetry: Arc<Telemetry>,
}

impl Runtime {
    pub fn new() -> Self {
        let registry = Arc::new(RwLock::new(ModelRegistry::new()));
        let telemetry = Arc::new(Telemetry::new());
        let manager = Arc::new(ModelManager::new(registry.clone(), telemetry.clone()));
        let router = Arc::new(ModelRouter::new(registry.clone()));
        
        Self {
            registry,
            manager,
            router,
            telemetry,
        }
    }

    pub async fn infer(&self, mut request: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        request.request_id = Some(request_id.clone());
        
        self.telemetry.record_request_start(&request_id).await;
        
        // Route to appropriate model
        let model_id = self.router.route(&request).await?;
        
        // Ensure model is loaded
        self.manager.ensure_loaded(&model_id).await?;
        
        // Execute inference
        let response = self.manager.infer(&model_id, &request).await?;
        
        self.telemetry.record_request_end(&request_id, &model_id).await;
        
        Ok(response)
    }

    pub async fn list_models(&self) -> Vec<crate::models::registry::ModelInfo> {
        self.registry.read().await.list_models()
    }

    pub async fn get_status(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "running",
            "models": self.list_models().await.len(),
        })
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
