use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Frequency-based next-model predictor (Phase 7 baseline).
///
/// Tracks bigram transitions (previous model -> next model counts) with a
/// unigram fallback. `predict` returns the most frequent successor of the
/// last observed model, or the overall most frequent model when the last
/// model has no recorded successors. Ties break by model id so results
/// are deterministic.
///
/// Hit counting is explicit: callers snapshot [`Self::predict`] before
/// serving a request, then call [`Self::record_outcome`] with the model
/// actually demanded. The runtime mirrors each outcome into
/// [`crate::telemetry::Telemetry::record_prefetch_outcome`] so the
/// prefetch hit rate is visible in telemetry snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextModelPredictor {
    transitions: HashMap<String, HashMap<String, u64>>,
    overall: HashMap<String, u64>,
    #[serde(skip, default)]
    last: Option<String>,
    history: VecDeque<String>,
    history_limit: usize,
    hits: u64,
    total_predictions: u64,
}

impl Default for NextModelPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl NextModelPredictor {
    pub fn new() -> Self {
        Self::with_history_limit(100)
    }

    pub fn with_history_limit(history_limit: usize) -> Self {
        Self {
            transitions: HashMap::new(),
            overall: HashMap::new(),
            last: None,
            history: VecDeque::new(),
            history_limit: history_limit.max(1),
            hits: 0,
            total_predictions: 0,
        }
    }

    pub fn history_limit(&self) -> usize {
        self.history_limit
    }

    pub fn last(&self) -> Option<&str> {
        self.last.as_deref()
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn total_predictions(&self) -> u64 {
        self.total_predictions
    }

    pub fn hit_rate(&self) -> Option<f64> {
        if self.total_predictions == 0 {
            None
        } else {
            Some(self.hits as f64 / self.total_predictions as f64)
        }
    }

    /// Observe the model actually served. Updates bigram + unigram counts
    /// and appends to the bounded history window.
    pub fn observe(&mut self, model_id: &str) {
        if let Some(prev) = self.last.clone() {
            *self
                .transitions
                .entry(prev)
                .or_default()
                .entry(model_id.to_string())
                .or_insert(0) += 1;
        }
        self.last = Some(model_id.to_string());
        *self.overall.entry(model_id.to_string()).or_insert(0) += 1;
        self.history.push_back(model_id.to_string());
        while self.history.len() > self.history_limit {
            self.history.pop_front();
        }
    }

    /// Predict the next model after the last observed one.
    pub fn predict(&self) -> Option<String> {
        match self.last.as_deref() {
            Some(prev) => self.predict_after(prev),
            None => most_frequent(&self.overall),
        }
    }

    /// Predict the successor of `prev`, falling back to the overall most
    /// frequent model when `prev` has no recorded successors.
    pub fn predict_after(&self, prev: &str) -> Option<String> {
        if let Some(nexts) = self.transitions.get(prev) {
            if let Some(best) = most_frequent(nexts) {
                return Some(best);
            }
        }
        most_frequent(&self.overall)
    }

    /// Score one prediction against the model actually demanded next.
    /// Returns true on a hit.
    pub fn record_outcome(&mut self, predicted: Option<&str>, actual: &str) -> bool {
        self.total_predictions += 1;
        let hit = predicted == Some(actual);
        if hit {
            self.hits += 1;
        }
        hit
    }

    pub fn reset(&mut self) {
        self.transitions.clear();
        self.overall.clear();
        self.last = None;
        self.history.clear();
        self.hits = 0;
        self.total_predictions = 0;
    }
}

fn most_frequent(counts: &HashMap<String, u64>) -> Option<String> {
    let mut best: Option<(&String, u64)> = None;
    for (model, count) in counts {
        let replace = match &best {
            None => true,
            Some((id, best_count)) => {
                (*count, std::cmp::Reverse(model.as_str()))
                    > (*best_count, std::cmp::Reverse(id.as_str()))
            }
        };
        if replace {
            best = Some((model, *count));
        }
    }
    best.map(|(model, _)| model.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_most_frequent_successor() {
        let mut predictor = NextModelPredictor::new();
        assert_eq!(predictor.predict(), None);
        for model in ["a", "b", "a", "b", "a", "c"] {
            predictor.observe(model);
        }
        // After "c" (no successors) fall back to overall most frequent: "a".
        assert_eq!(predictor.predict().as_deref(), Some("a"));
        assert_eq!(predictor.predict_after("a").as_deref(), Some("b"));
        assert_eq!(predictor.predict_after("b").as_deref(), Some("a"));
    }

    #[test]
    fn ties_break_by_id() {
        let mut predictor = NextModelPredictor::new();
        predictor.observe("b");
        predictor.observe("a");
        // Overall counts tied at 1; smallest id wins.
        assert_eq!(predictor.predict_after("ghost").as_deref(), Some("a"));
    }

    #[test]
    fn hit_counting_tracks_accuracy() {
        let mut predictor = NextModelPredictor::new();
        assert_eq!(predictor.hit_rate(), None);
        predictor.observe("a");
        predictor.observe("b");
        let predicted = predictor.predict_after("a");
        assert!(predictor.record_outcome(predicted.as_deref(), "b"));
        assert!(!predictor.record_outcome(predicted.as_deref(), "c"));
        assert_eq!(predictor.hits(), 1);
        assert_eq!(predictor.total_predictions(), 2);
        assert!((predictor.hit_rate().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn history_window_is_bounded() {
        let mut predictor = NextModelPredictor::with_history_limit(3);
        for model in ["a", "b", "c", "d"] {
            predictor.observe(model);
        }
        assert_eq!(predictor.len(), 3);
        assert_eq!(predictor.last(), Some("d"));
    }
}
