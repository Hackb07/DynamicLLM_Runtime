use crate::core::error::Result;
use crate::core::types::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;
use futures_util::stream::Stream;
use std::pin::Pin;

/// A stream of generated text chunks. Errors end the stream after the
/// handler surfaces them (see the SSE endpoint).
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Common interface implemented by every inference backend.
#[async_trait]
pub trait Adapter: Send + Sync {
    /// Human-readable adapter name (e.g. `"openai"`).
    fn name(&self) -> &str;

    /// Backend key matching `ModelInfo::backend` (e.g. `"llama.cpp"`).
    fn backend(&self) -> &str;

    /// Run one non-streaming inference request to completion.
    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse>;

    /// Run a streaming inference request, yielding text deltas.
    /// The default implementation buffers one full response as a single chunk.
    async fn infer_stream(&self, request: &InferenceRequest) -> Result<ChunkStream> {
        let response = self.infer(request).await?;
        let chunk = response.content.clone();
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(chunk) },
        )))
    }

    /// Prepare the backend for inference. Called by `ModelManager` during
    /// `ensure_loaded`. Local backends override this to verify reachability;
    /// cloud backends use the default no-op.
    async fn load(&self) -> Result<()> {
        Ok(())
    }

    /// Release backend resources. Called by `ModelManager` during `unload`.
    /// Managed local backends override this to stop processes they started;
    /// remote/cloud backends use the default no-op.
    async fn unload(&self) -> Result<()> {
        Ok(())
    }

    /// Best-effort reachability check. Returns `false` (not an error) when
    /// the backend is unreachable.
    async fn health_check(&self) -> Result<bool>;
}
