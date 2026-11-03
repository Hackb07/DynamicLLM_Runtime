use crate::core::error::{Error, Result};
use crate::core::types::InferenceRequest;
use crate::models::registry::{ModelInfo, ModelRegistry};
use crate::routing::strategy::{score_candidate, RoutingStrategy, RoutingWeights};
use crate::telemetry::Telemetry;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelRouter {
    registry: Arc<RwLock<ModelRegistry>>,
    strategy: RoutingStrategy,
    fixed_model: Option<String>,
    telemetry: Option<Arc<Telemetry>>,
    weights: RoutingWeights,
}

impl ModelRouter {
    pub fn new(registry: Arc<RwLock<ModelRegistry>>) -> Self {
        Self {
            registry,
            strategy: RoutingStrategy::default(),
            fixed_model: None,
            telemetry: None,
            weights: RoutingWeights::default(),
        }
    }

    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_fixed_model(mut self, model: impl Into<String>) -> Self {
        self.fixed_model = Some(model.into());
        self
    }

    /// Attach telemetry for adaptive/quality-aware scoring. Without it,
    /// those strategies fall back to deterministic id ordering.
    pub fn with_telemetry(mut self, telemetry: Arc<Telemetry>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    pub fn with_weights(mut self, weights: RoutingWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    pub fn weights(&self) -> RoutingWeights {
        self.weights
    }

    /// Primary choice: the first of [`ModelRouter::route_candidates`].
    pub async fn route(&self, request: &InferenceRequest) -> Result<String> {
        let candidates = self.route_candidates(request).await?;
        candidates
            .into_iter()
            .next()
            .ok_or_else(|| Error::Routing("No available models".to_string()))
    }

    /// Ordered eligible model ids. An explicit `request.model` always wins
    /// (validated against the registry); otherwise candidates are filtered
    /// by capability and ordered by strategy. Deterministic: ties break by id.
    pub async fn route_candidates(&self, request: &InferenceRequest) -> Result<Vec<String>> {
        let registry = self.registry.read().await;

        if let Some(model) = &request.model {
            registry.get(model)?;
            return Ok(vec![model.clone()]);
        }

        if self.strategy == RoutingStrategy::Fixed {
            let fixed = self.fixed_model.as_ref().ok_or_else(|| {
                Error::Routing("fixed strategy configured without a model".to_string())
            })?;
            registry.get(fixed)?;
            return Ok(vec![fixed.clone()]);
        }

        // An empty tools vec means "no tool requirement", same as None.
        let needs_tools = request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());
        let mut candidates: Vec<ModelInfo> = registry
            .list_models()
            .into_iter()
            .filter(|model| model.enabled)
            .filter(|model| !needs_tools || model.capabilities.supports_tools)
            .filter(|model| request.stream != Some(true) || model.capabilities.supports_streaming)
            .collect();

        if candidates.is_empty() {
            return Err(Error::Routing(
                "No enabled model satisfies the request capabilities".to_string(),
            ));
        }

        if matches!(
            self.strategy,
            RoutingStrategy::RuleBased
                | RoutingStrategy::CapabilityBased
                | RoutingStrategy::Fallback
        ) {
            if let Some(task) = request.metadata.as_ref().and_then(|m| m.get("task")) {
                let matching: Vec<ModelInfo> = candidates
                    .iter()
                    .filter(|model| {
                        model
                            .capabilities
                            .task_affinity
                            .iter()
                            .any(|affinity| affinity == task)
                    })
                    .cloned()
                    .collect();
                if matching.is_empty() {
                    tracing::warn!(
                        "No model has task affinity for '{task}'; falling back to capability match"
                    );
                } else {
                    candidates = matching;
                }
            }
        }

        match self.strategy {
            RoutingStrategy::ResourceAware => {
                candidates.sort_by_key(|model| (model.memory_required_mb, model.id.clone()));
            }
            RoutingStrategy::Adaptive | RoutingStrategy::QualityAware => {
                candidates = self.order_by_quality(candidates, request).await;
            }
            _ => {
                candidates.sort_by_key(|model| model.id.clone());
            }
        }

        Ok(candidates.into_iter().map(|model| model.id).collect())
    }

    /// Order capability-eligible candidates by measured telemetry blended
    /// with a static task-affinity bonus. Task affinity is a soft bonus
    /// here (not the hard filter used by rule-based routing) so a fast,
    /// reliable generalist can still win over a slow specialist.
    async fn order_by_quality(
        &self,
        candidates: Vec<ModelInfo>,
        request: &InferenceRequest,
    ) -> Vec<ModelInfo> {
        let Some(telemetry) = self.telemetry.as_ref() else {
            tracing::warn!("telemetry-scored routing without telemetry; falling back to id order");
            let mut fallback = candidates;
            fallback.sort_by_key(|model| model.id.clone());
            return fallback;
        };
        let task = request.metadata.as_ref().and_then(|m| m.get("task"));
        let mut scored = Vec::with_capacity(candidates.len());
        for model in candidates {
            let stats = telemetry.model_stats(&model.id).await;
            let affinity_match = task
                .map(|t| model.capabilities.task_affinity.iter().any(|a| a == t))
                .unwrap_or(false);
            scored.push((model, stats, affinity_match));
        }
        let max_latency = scored
            .iter()
            .filter_map(|(_, stats, _)| stats.avg_latency_ms)
            .fold(None::<f64>, |max, v| {
                Some(max.map(|m: f64| m.max(v)).unwrap_or(v))
            });
        scored.sort_by(|a, b| {
            let sa = score_candidate(&a.1, a.2, max_latency, &self.weights);
            let sb = score_candidate(&b.1, b.2, max_latency, &self.weights);
            // Higher score first; NaN is impossible (weights are finite,
            // validated at config load) but break ties by id regardless.
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        scored.into_iter().map(|(model, _, _)| model).collect()
    }
}
