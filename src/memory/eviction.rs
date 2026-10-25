#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    CostAware,
    PriorityAware,
}
