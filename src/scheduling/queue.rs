use crate::core::error::{Error, Result};
use tokio::sync::{mpsc, Mutex};

/// Small bounded FIFO used for staged work (e.g. pending agent steps).
/// Pushers apply backpressure instead of growing memory without bound.
pub struct RequestQueue<T> {
    capacity: usize,
    tx: mpsc::Sender<T>,
    rx: Mutex<mpsc::Receiver<T>>,
}

impl<T> RequestQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        Self {
            capacity: capacity.max(1),
            tx,
            rx: Mutex::new(rx),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Enqueue without waiting; fails when the queue is full.
    pub fn try_push(&self, item: T) -> Result<()> {
        self.tx
            .try_send(item)
            .map_err(|_| Error::ResourceUnavailable("request queue is full".to_string()))
    }

    /// Enqueue, waiting for space.
    pub async fn push(&self, item: T) -> Result<()> {
        self.tx.send(item).await.map_err(|_| Error::Cancelled)
    }

    /// Dequeue the next item; returns `None` once all senders are gone.
    pub async fn pop(&self) -> Option<T> {
        self.rx.lock().await.recv().await
    }

    pub fn len(&self) -> usize {
        self.tx.max_capacity() - self.tx.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.tx.capacity() == self.tx.max_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fifo_order_and_backpressure() {
        let queue = RequestQueue::new(2);
        assert!(queue.is_empty());
        queue.try_push(1).unwrap();
        queue.try_push(2).unwrap();
        assert!(queue.try_push(3).is_err());
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop().await, Some(1));
        assert_eq!(queue.pop().await, Some(2));
        assert!(queue.is_empty());
    }
}
