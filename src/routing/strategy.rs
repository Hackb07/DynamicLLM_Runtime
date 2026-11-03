use crate::telemetry::ModelStats;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Always use one configured model id.
    Fixed,
    /// Task-affinity mapping with capability filtering.
    #[default]
    RuleBased,
    /// Filter strictly by required capabilities, then task affinity.
    CapabilityBased,
    /// Among eligible models, prefer the smallest declared footprint.
    ResourceAware,
    /// Deterministic id ordering (same as rule-based); breadth comes from
    /// the runtime failing over across the first `max_fallback_models`
    /// candidates rather than from the ordering itself.
    Fallback,
    /// Score eligible models from measured telemetry (latency + success
    /// rate) blended with a static task-affinity bonus. Falls back to id
    /// order when no telemetry is attached.
    Adaptive,
    /// Same scoring as adaptive; kept as a distinct name for the
    /// README "quality-aware" policy so configs can select either.
    QualityAware,
}

impl RoutingStrategy {
    /// Parse a config string, defaulting to rule-based on unknown input.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "fixed" => Self::Fixed,
            "capability_based" | "capability-based" | "capability" => Self::CapabilityBased,
            "resource_aware" | "resource-aware" | "resource" => Self::ResourceAware,
            "fallback" => Self::Fallback,
            "adaptive" => Self::Adaptive,
            "quality_aware" | "quality-aware" | "quality" => Self::QualityAware,
            _ => Self::RuleBased,
        }
    }

    pub fn is_telemetry_scored(self) -> bool {
        matches!(self, Self::Adaptive | Self::QualityAware)
    }
}

/// Configurable weights for telemetry-scored routing. All fields have
/// serde defaults so configs written before Phase 7 still parse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoutingWeights {
    #[serde(default = "default_latency_weight")]
    pub latency_weight: f64,
    #[serde(default = "default_success_weight")]
    pub success_weight: f64,
    /// Bonus added when the candidate matches the request task affinity.
    #[serde(default = "default_affinity_bonus")]
    pub affinity_bonus: f64,
}

fn default_latency_weight() -> f64 {
    1.0
}

fn default_success_weight() -> f64 {
    1.0
}

fn default_affinity_bonus() -> f64 {
    0.5
}

impl Default for RoutingWeights {
    fn default() -> Self {
        Self {
            latency_weight: default_latency_weight(),
            success_weight: default_success_weight(),
            affinity_bonus: default_affinity_bonus(),
        }
    }
}

impl RoutingWeights {
    pub fn new(latency_weight: f64, success_weight: f64, affinity_bonus: f64) -> Self {
        Self {
            latency_weight,
            success_weight,
            affinity_bonus,
        }
    }
}

/// Score one candidate; higher is better.
///
/// - `success_rate`: measured ok/(ok+err), or a neutral 0.5 prior when
///   the model has no samples yet (so new models rank between proven
///   good and proven bad instead of at either extreme).
/// - `latency`: normalized against the slowest candidate with samples:
///   `1 - avg/max`, so the fastest scores near 1 and the slowest near 0;
///   models without samples get a neutral 0.5.
/// - `affinity_match`: static task-affinity bonus, preserving the
///   capability/task signal alongside measured quality.
pub fn score_candidate(
    stats: &ModelStats,
    affinity_match: bool,
    max_latency_ms: Option<f64>,
    weights: &RoutingWeights,
) -> f64 {
    let success = stats.success_rate.unwrap_or(0.5);
    let latency_term = match (stats.avg_latency_ms, max_latency_ms) {
        (Some(avg), Some(max)) if max > 0.0 => 1.0 - (avg / max).clamp(0.0, 1.0),
        _ => 0.5,
    };
    weights.success_weight * success
        + weights.latency_weight * latency_term
        + if affinity_match {
            weights.affinity_bonus
        } else {
            0.0
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_aliases_and_defaults_to_rule_based() {
        assert_eq!(RoutingStrategy::parse("fixed"), RoutingStrategy::Fixed);
        assert_eq!(
            RoutingStrategy::parse("resource-aware"),
            RoutingStrategy::ResourceAware
        );
        assert_eq!(
            RoutingStrategy::parse("adaptive"),
            RoutingStrategy::Adaptive
        );
        assert_eq!(
            RoutingStrategy::parse("quality-aware"),
            RoutingStrategy::QualityAware
        );
        assert_eq!(
            RoutingStrategy::parse("quality_aware"),
            RoutingStrategy::QualityAware
        );
        assert_eq!(RoutingStrategy::parse("bogus"), RoutingStrategy::RuleBased);
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::RuleBased);
    }

    #[test]
    fn weights_default_for_old_configs() {
        let weights: RoutingWeights = serde_json::from_str("{}").unwrap();
        assert_eq!(weights, RoutingWeights::default());
    }

    #[test]
    fn faster_and_reliable_scores_highest() {
        let weights = RoutingWeights::default();
        let fast_good = ModelStats {
            avg_latency_ms: Some(100.0),
            success_rate: Some(1.0),
            samples: 10,
        };
        let slow_bad = ModelStats {
            avg_latency_ms: Some(1000.0),
            success_rate: Some(0.2),
            samples: 10,
        };
        let max = Some(1000.0);
        assert!(
            score_candidate(&fast_good, false, max, &weights)
                > score_candidate(&slow_bad, false, max, &weights)
        );
        // Task affinity can only help, never hurt.
        assert!(
            score_candidate(&slow_bad, true, max, &weights)
                > score_candidate(&slow_bad, false, max, &weights)
        );
    }

    #[test]
    fn unknown_models_get_neutral_prior() {
        let weights = RoutingWeights::default();
        let unknown = ModelStats {
            avg_latency_ms: None,
            success_rate: None,
            samples: 0,
        };
        let perfect = ModelStats {
            avg_latency_ms: Some(50.0),
            success_rate: Some(1.0),
            samples: 5,
        };
        let terrible = ModelStats {
            avg_latency_ms: Some(5000.0),
            success_rate: Some(0.0),
            samples: 5,
        };
        let max = Some(5000.0);
        let mid = score_candidate(&unknown, false, max, &weights);
        assert!(score_candidate(&perfect, false, max, &weights) > mid);
        assert!(mid > score_candidate(&terrible, false, max, &weights));
    }
}
