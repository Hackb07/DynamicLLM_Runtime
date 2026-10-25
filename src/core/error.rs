use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not ready: {0}")]
    ModelNotReady(String),

    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Routing error: {0}")]
    Routing(String),

    #[error("Timeout")]
    Timeout,

    #[error("Cancelled")]
    Cancelled,

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
