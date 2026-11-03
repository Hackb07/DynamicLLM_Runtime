use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Heuristic memory estimates for admission decisions.
///
/// These are intentionally coarse: exact residency depends on the backend,
/// quantization, and batch size. Estimates feed the manager's budget
/// accounting, not a precise allocator.
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Weights plus a linear KV-cache allowance (~0.5 MB per context token).
    pub fn estimate(weights_mb: u64, context_length: usize) -> u64 {
        weights_mb.saturating_add((context_length as u64).saturating_mul(1) / 2)
    }

    /// Whether `required_mb` fits into `available_mb`.
    /// Zero requirement (cloud models) always fits.
    pub fn fits(required_mb: u64, available_mb: u64) -> bool {
        required_mb == 0 || required_mb <= available_mb
    }
}

/// Per-model residency: assumed estimate vs measured footprint.
///
/// Measured-vs-assumed behavior (Phase 7):
/// - `estimated_mb` is ASSUMED: the heuristic from config
///   (`memory_required_mb` / [`MemoryEstimator::estimate`]). It is always
///   available and is what admission/eviction budgets use today.
/// - `measured_mb` is MEASURED: a backend- or host-observed footprint, set
///   only when a backend reports one (currently no backend does, so this
///   stays `None` unless tests or future adapters record it). It never
///   overrides admission by itself; [`ResidencyTracker::effective_mb`]
///   prefers it only for observability and future cost-aware policies.
/// - `resident` tracks whether the manager considers the model loaded.
///   Eviction hooks ([`ResidencyTracker::on_loaded`] /
///   [`ResidencyTracker::on_evicted`]) keep this in sync with lifecycle
///   transitions; estimates are retained across evictions so the next
///   admission still has an assumed size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyInfo {
    pub model_id: String,
    pub estimated_mb: u64,
    pub measured_mb: Option<u64>,
    pub resident: bool,
}

impl ResidencyInfo {
    pub fn new(model_id: impl Into<String>, estimated_mb: u64) -> Self {
        Self {
            model_id: model_id.into(),
            estimated_mb,
            measured_mb: None,
            resident: false,
        }
    }

    /// Footprint to display/plan with: measured when a backend reported
    /// one, otherwise the assumed estimate. Admission budgets today use
    /// the lifecycle's assumed `memory_mb`; this helper exists so future
    /// policies can switch to measured values without changing callers.
    pub fn effective_mb(&self) -> u64 {
        self.measured_mb.unwrap_or(self.estimated_mb)
    }
}

/// Tracks [`ResidencyInfo`] per model. Owns the eviction-hook methods the
/// manager calls on load/unload; also implements
/// [`crate::memory::eviction::EvictionListener`] so eviction code can
/// notify residency through the trait.
#[derive(Debug, Clone, Default)]
pub struct ResidencyTracker {
    models: HashMap<String, ResidencyInfo>,
}

impl ResidencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, model_id: &str) -> Option<&ResidencyInfo> {
        self.models.get(model_id)
    }

    pub fn set_estimated(&mut self, model_id: &str, estimated_mb: u64) {
        self.models
            .entry(model_id.to_string())
            .or_insert_with(|| ResidencyInfo::new(model_id, estimated_mb))
            .estimated_mb = estimated_mb;
    }

    pub fn set_measured(&mut self, model_id: &str, measured_mb: u64) {
        self.models
            .entry(model_id.to_string())
            .or_insert_with(|| ResidencyInfo::new(model_id, 0))
            .measured_mb = Some(measured_mb);
    }

    pub fn clear_measured(&mut self, model_id: &str) {
        if let Some(info) = self.models.get_mut(model_id) {
            info.measured_mb = None;
        }
    }

    pub fn mark_resident(&mut self, model_id: &str, resident: bool) {
        self.models
            .entry(model_id.to_string())
            .or_insert_with(|| ResidencyInfo::new(model_id, 0))
            .resident = resident;
    }

    /// Eviction hook: the manager calls this after a load succeeds.
    /// Records the assumed size and marks the model resident. A stale
    /// measured value is kept (it describes this model, not this load).
    pub fn on_loaded(&mut self, model_id: &str, assumed_mb: u64) {
        let info = self
            .models
            .entry(model_id.to_string())
            .or_insert_with(|| ResidencyInfo::new(model_id, assumed_mb));
        info.estimated_mb = assumed_mb;
        info.resident = true;
    }

    /// Eviction hook: the manager calls this after an unload/eviction.
    /// Clears residency but retains estimates for the next admission.
    pub fn on_evicted(&mut self, model_id: &str) {
        if let Some(info) = self.models.get_mut(model_id) {
            info.resident = false;
        } else {
            let mut info = ResidencyInfo::new(model_id, 0);
            info.resident = false;
            self.models.insert(model_id.to_string(), info);
        }
    }

    pub fn effective_mb(&self, model_id: &str) -> u64 {
        self.models
            .get(model_id)
            .map(ResidencyInfo::effective_mb)
            .unwrap_or(0)
    }

    pub fn resident_models(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .models
            .values()
            .filter(|info| info.resident)
            .map(|info| info.model_id.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Sum of effective footprints for resident models. Mixes measured
    /// (where available) with assumed values; see struct docs.
    pub fn resident_mb(&self) -> u64 {
        self.models
            .values()
            .filter(|info| info.resident)
            .map(ResidencyInfo::effective_mb)
            .sum()
    }

    pub fn snapshot(&self) -> Vec<ResidencyInfo> {
        let mut infos: Vec<ResidencyInfo> = self.models.values().cloned().collect();
        infos.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        infos
    }
}

impl crate::memory::eviction::EvictionListener for ResidencyTracker {
    fn on_evict(&mut self, model_id: &str) {
        self.on_evicted(model_id);
    }

    fn on_load(&mut self, model_id: &str, memory_mb: u64) {
        self.on_loaded(model_id, memory_mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_adds_context_allowance() {
        assert_eq!(MemoryEstimator::estimate(2000, 4096), 2000 + 2048);
    }

    #[test]
    fn zero_requirement_always_fits() {
        assert!(MemoryEstimator::fits(0, 0));
        assert!(MemoryEstimator::fits(100, 100));
        assert!(!MemoryEstimator::fits(101, 100));
    }

    #[test]
    fn measured_overrides_assumed_for_effective() {
        let mut tracker = ResidencyTracker::new();
        tracker.on_loaded("m1", 2048);
        assert_eq!(tracker.effective_mb("m1"), 2048);
        tracker.set_measured("m1", 3072);
        assert_eq!(tracker.effective_mb("m1"), 3072);
        tracker.clear_measured("m1");
        assert_eq!(tracker.effective_mb("m1"), 2048);
    }

    #[test]
    fn eviction_hooks_track_residency() {
        let mut tracker = ResidencyTracker::new();
        tracker.on_loaded("a", 100);
        tracker.on_loaded("b", 200);
        assert_eq!(tracker.resident_models(), vec!["a", "b"]);
        assert_eq!(tracker.resident_mb(), 300);
        tracker.on_evicted("a");
        assert_eq!(tracker.resident_models(), vec!["b"]);
        // Estimates survive eviction for the next admission.
        assert_eq!(tracker.get("a").unwrap().estimated_mb, 100);
        assert!(!tracker.get("a").unwrap().resident);
    }
}
