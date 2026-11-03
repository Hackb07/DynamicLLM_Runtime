use crate::models::lifecycle::ModelLifecycle;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    /// Evict the model unused for the longest time.
    #[default]
    Lru,
    /// Evict the model with the fewest served requests.
    Lfu,
    /// Evict the largest resident model first (frees the most budget).
    CostAware,
    /// Evict the lowest-priority model first (ties broken by age).
    PriorityAware,
}

impl EvictionPolicy {
    /// Parse a config string, defaulting to LRU on unknown input.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "lfu" | "least_frequently_used" => Self::Lfu,
            "cost_aware" | "cost-aware" | "costaware" | "cost" => Self::CostAware,
            "priority_aware" | "priority-aware" | "priorityaware" | "priority" => {
                Self::PriorityAware
            }
            _ => Self::Lru,
        }
    }

    /// Pick one eviction victim among resident-model lifecycles.
    /// Models never used (`last_used == None`) count as the oldest.
    pub fn pick<'a>(
        &self,
        candidates: impl IntoIterator<Item = &'a ModelLifecycle>,
    ) -> Option<String> {
        let mut best: Option<&'a ModelLifecycle> = None;
        for candidate in candidates {
            let replace = match best {
                None => true,
                Some(current) => match self {
                    Self::Lru => {
                        (candidate.last_used, &candidate.model_id)
                            < (current.last_used, &current.model_id)
                    }
                    Self::Lfu => {
                        (candidate.use_count, &candidate.model_id)
                            < (current.use_count, &current.model_id)
                    }
                    Self::CostAware => {
                        (candidate.memory_mb, &candidate.model_id)
                            > (current.memory_mb, &current.model_id)
                    }
                    Self::PriorityAware => {
                        (candidate.priority, candidate.last_used, &candidate.model_id)
                            < (current.priority, current.last_used, &current.model_id)
                    }
                },
            };
            if replace {
                best = Some(candidate);
            }
        }
        best.map(|lifecycle| lifecycle.model_id.clone())
    }
}

/// Hook notified on load/eviction so residency trackers stay in sync
/// with lifecycle transitions. The manager calls these hooks; the
/// default [`crate::memory::estimator::ResidencyTracker`] implements
/// this trait. Measured-vs-assumed note: hooks carry the ASSUMED
/// footprint (what admission budgeted). Measured values arrive
/// separately via `set_measured` when a backend reports them.
pub trait EvictionListener {
    fn on_evict(&mut self, model_id: &str);
    fn on_load(&mut self, model_id: &str, memory_mb: u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn lifecycle(
        id: &str,
        use_count: u64,
        memory_mb: u64,
        priority: u32,
        age_secs: Option<u64>,
    ) -> ModelLifecycle {
        let mut lifecycle = ModelLifecycle::new(id);
        lifecycle.use_count = use_count;
        lifecycle.memory_mb = memory_mb;
        lifecycle.priority = priority;
        lifecycle.last_used = age_secs.map(|secs| Instant::now() - Duration::from_secs(secs));
        lifecycle
    }

    #[test]
    fn lru_picks_oldest_and_never_used_first() {
        let models = [
            lifecycle("new", 10, 100, 0, Some(1)),
            lifecycle("old", 10, 100, 0, Some(100)),
            lifecycle("fresh", 0, 100, 0, None),
        ];
        assert_eq!(
            EvictionPolicy::Lru.pick(models.iter()),
            Some("fresh".to_string())
        );

        let models = [
            lifecycle("new", 10, 100, 0, Some(1)),
            lifecycle("old", 10, 100, 0, Some(100)),
        ];
        assert_eq!(
            EvictionPolicy::Lru.pick(models.iter()),
            Some("old".to_string())
        );
    }

    #[test]
    fn lfu_picks_least_used() {
        let models = [
            lifecycle("a", 5, 100, 0, Some(1)),
            lifecycle("b", 1, 100, 0, Some(1)),
        ];
        assert_eq!(
            EvictionPolicy::Lfu.pick(models.iter()),
            Some("b".to_string())
        );
    }

    #[test]
    fn cost_aware_picks_largest() {
        let models = [
            lifecycle("small", 5, 512, 0, Some(1)),
            lifecycle("big", 5, 4096, 0, Some(1)),
        ];
        assert_eq!(
            EvictionPolicy::CostAware.pick(models.iter()),
            Some("big".to_string())
        );
    }

    #[test]
    fn priority_aware_picks_lowest_priority() {
        let models = [
            lifecycle("hi", 1, 100, 9, Some(100)),
            lifecycle("lo", 99, 100, 1, Some(1)),
        ];
        assert_eq!(
            EvictionPolicy::PriorityAware.pick(models.iter()),
            Some("lo".to_string())
        );
    }

    #[test]
    fn pick_is_none_without_candidates() {
        assert_eq!(
            EvictionPolicy::Lru.pick(std::iter::empty::<&ModelLifecycle>()),
            None
        );
    }

    #[test]
    fn parse_accepts_aliases_and_defaults_to_lru() {
        assert_eq!(EvictionPolicy::parse("lfu"), EvictionPolicy::Lfu);
        assert_eq!(
            EvictionPolicy::parse("cost-aware"),
            EvictionPolicy::CostAware
        );
        assert_eq!(
            EvictionPolicy::parse("priority_aware"),
            EvictionPolicy::PriorityAware
        );
        assert_eq!(EvictionPolicy::parse("lru"), EvictionPolicy::Lru);
        assert_eq!(EvictionPolicy::parse("bogus"), EvictionPolicy::Lru);
    }
}
