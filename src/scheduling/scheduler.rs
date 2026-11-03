use crate::core::error::{Error, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Concurrency gate: bounds simultaneous inference requests so one burst
/// cannot exhaust backend connections or memory. Callers that arrive while
/// all permits are held wait up to `wait_timeout`, then fail fast with
/// `ResourceUnavailable` instead of queueing forever.
pub struct Scheduler {
    semaphore: Arc<Semaphore>,
    wait_timeout: Duration,
}

impl Scheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent.max(1))),
            wait_timeout: Duration::from_secs(10),
        }
    }

    pub fn with_wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit> {
        match self.semaphore.clone().try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(_) => {
                tokio::time::timeout(self.wait_timeout, self.semaphore.clone().acquire_owned())
                    .await
                    .map_err(|_| {
                        Error::ResourceUnavailable(
                            "concurrency limit reached; server is busy".to_string(),
                        )
                    })?
                    .map_err(|_| Error::Cancelled)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn permits_are_limited_and_released() {
        let scheduler = Scheduler::new(1).with_wait_timeout(Duration::from_millis(50));
        let _first = scheduler.acquire().await.unwrap();
        assert!(scheduler.acquire().await.is_err());
        drop(_first);
        assert!(scheduler.acquire().await.is_ok());
    }
}
