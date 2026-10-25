#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    Fixed,
    RuleBased,
    CapabilityBased,
    ResourceAware,
    Fallback,
}
