use crate::core::error::{Error, Result};
use crate::models::capabilities::ModelCapabilities;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub capabilities: ModelCapabilities,
    pub context_limit: usize,
    pub memory_required_mb: u64,
    pub quantization: Option<String>,
    pub enabled: bool,
}

#[derive(Debug)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.register_defaults();
        registry
    }

    fn register_defaults(&mut self) {
        self.register(ModelInfo {
            id: "gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            backend: "openai".to_string(),
            capabilities: ModelCapabilities {
                supports_tools: true,
                supports_streaming: true,
                supports_structured_output: true,
                task_affinity: vec!["general".to_string(), "qa".to_string()],
            },
            context_limit: 4096,
            memory_required_mb: 0,
            quantization: None,
            enabled: true,
        });

        self.register(ModelInfo {
            id: "llama-3.2-3b".to_string(),
            name: "LLaMA 3.2 3B".to_string(),
            backend: "llama.cpp".to_string(),
            capabilities: ModelCapabilities {
                supports_tools: false,
                supports_streaming: true,
                supports_structured_output: false,
                task_affinity: vec!["summarization".to_string(), "general".to_string()],
            },
            context_limit: 8192,
            memory_required_mb: 2048,
            quantization: Some("GGUF".to_string()),
            enabled: true,
        });
    }

    pub fn register(&mut self, model: ModelInfo) {
        self.models.insert(model.id.clone(), model);
    }

    pub fn get(&self, id: &str) -> Result<ModelInfo> {
        self.models.get(id).cloned().ok_or_else(|| Error::ModelNotFound(id.to_string()))
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.values().cloned().collect()
    }

    pub fn find_eligible(&self, task_affinity: &str) -> Vec<ModelInfo> {
        self.models
            .values()
            .filter(|m| m.enabled && m.capabilities.task_affinity.contains(&task_affinity.to_string()))
            .cloned()
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
