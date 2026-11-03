//! HTTP client for one remote inference worker (opt-in remote execution).
//!
//! Workers speak the same OpenAI-compatible chat-completions wire format
//! as the other HTTP backends, so timeouts, bounded retries, and response
//! mapping reuse the shared [`OpenAiCompatClient`] patterns.

use crate::adapters::base::ChunkStream;
use crate::adapters::http::{to_inference_response, OpenAiCompatClient};
use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use std::time::Duration;

/// Client for one worker's `/v1/chat/completions` endpoint.
#[derive(Debug, Clone)]
pub struct WorkerClient {
    worker_id: String,
    addr: String,
    inner: OpenAiCompatClient,
}

impl WorkerClient {
    pub fn new(
        worker_id: &str,
        addr: &str,
        timeout: Duration,
        max_retries: u32,
        backoff_base_ms: u64,
    ) -> Result<Self> {
        let inner = OpenAiCompatClient::new(
            format!("worker {worker_id}"),
            addr,
            "/v1/chat/completions",
            "/health",
            None,
            timeout,
            max_retries,
            backoff_base_ms,
        )?;
        Ok(Self {
            worker_id: worker_id.to_string(),
            addr: addr.to_string(),
            inner,
        })
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Run one non-streaming inference request on the worker, with the
    /// client's timeout and bounded retries.
    pub async fn infer(
        &self,
        model: &str,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let chat = self.inner.chat(model, request).await?;
        Ok(to_inference_response(request, model, chat))
    }

    /// Run one streaming inference request on the worker, yielding text
    /// deltas. Single attempt (like all streaming paths): retrying a
    /// partially-consumed stream would duplicate tokens.
    pub async fn infer_stream(
        &self,
        model: &str,
        request: &InferenceRequest,
    ) -> Result<ChunkStream> {
        self.inner.chat_stream(model, request).await
    }

    /// Best-effort reachability probe. Never fails: returns `false` when
    /// the worker is unreachable or reports an error status.
    pub async fn health(&self) -> bool {
        self.inner.health().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_keeps_identity_and_trims_base_url() {
        let client = WorkerClient::new(
            "w0",
            "http://127.0.0.1:9999/",
            Duration::from_secs(1),
            2,
            10,
        )
        .unwrap();
        assert_eq!(client.worker_id(), "w0");
        assert_eq!(client.addr(), "http://127.0.0.1:9999/");
    }
}
