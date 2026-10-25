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

#[derive(Debug, Clone)]
pub struct ModelLifecycle {
    pub model_id: String,
    pub state: ModelState,
    pub load_count: u64,
    pub last_used: Option<std::time::Instant>,
}
