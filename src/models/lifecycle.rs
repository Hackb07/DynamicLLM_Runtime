use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelState {
    Registered,
    Unloaded,
    Loading,
    Ready,
    Busy,
    Evicting,
    Failed,
}

/// Resident-state bookkeeping for one model.
#[derive(Debug, Clone)]
pub struct ModelLifecycle {
    pub model_id: String,
    pub state: ModelState,
    pub load_count: u64,
    /// Successful inference calls served while resident.
    pub use_count: u64,
    /// Declared footprint (MB) currently counted against the local budget.
    pub memory_mb: u64,
    /// Higher survives longer under priority-aware eviction.
    pub priority: u32,
    pub last_used: Option<std::time::Instant>,
    pub last_error: Option<String>,
}

impl ModelLifecycle {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            state: ModelState::Unloaded,
            load_count: 0,
            use_count: 0,
            memory_mb: 0,
            priority: 0,
            last_used: None,
            last_error: None,
        }
    }
}
