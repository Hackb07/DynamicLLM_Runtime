pub struct MemoryEstimator;

impl MemoryEstimator {
    pub fn estimate(model_size_mb: u64, context_length: usize) -> u64 {
        model_size_mb + (context_length as u64 * 0.5)
    }
}
