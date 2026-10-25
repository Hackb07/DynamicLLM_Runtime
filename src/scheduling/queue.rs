pub struct RequestQueue {
    capacity: usize,
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}
