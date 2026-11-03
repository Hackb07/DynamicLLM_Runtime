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

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Client-safe message without the `Display` kind prefix.
    ///
    /// [`crate::api::handlers::ApiError`] and the SSE `error` event both use
    /// this so JSON errors and mid-stream errors report identical text.
    pub fn user_message(&self) -> String {
        match self {
            Error::Config(message)
            | Error::ModelNotFound(message)
            | Error::ModelNotReady(message)
            | Error::ResourceUnavailable(message)
            | Error::Backend(message)
            | Error::Routing(message)
            | Error::Validation(message)
            | Error::Forbidden(message)
            | Error::Other(message) => message.clone(),
            Error::Timeout => "request timed out".to_string(),
            Error::Cancelled => "cancelled".to_string(),
            Error::Io(e) => format!("io error: {e}"),
            Error::Serialization(e) => format!("serialization error: {e}"),
        }
    }
}
