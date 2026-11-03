pub mod base;
pub mod http;
pub mod llama_cpp;
pub mod local;
pub mod openai;
pub mod worker;

pub use base::{Adapter, ChunkStream};
pub use http::{to_inference_response, OpenAiCompatClient};
pub use llama_cpp::{server_args, LlamaCppAdapter, ManagedServerConfig};
pub use local::LocalEngineAdapter;
pub use openai::OpenAIAdapter;
pub use worker::WorkerClient;
