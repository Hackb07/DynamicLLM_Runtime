pub struct TaskController {
    max_steps: u32,
}

impl TaskController {
    pub fn new(max_steps: u32) -> Self {
        Self { max_steps }
    }
}
