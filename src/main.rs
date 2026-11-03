use dynamic_llm_runtime::config::{ModelsFile, RuntimeConfig};
use dynamic_llm_runtime::{api, core};
use std::sync::Arc;
use tokio::net::TcpListener;

fn log_level() -> tracing::level_filters::LevelFilter {
    let raw = std::env::var("DLLM_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| "info".to_string());
    match raw.to_ascii_lowercase().as_str() {
        "trace" => tracing::level_filters::LevelFilter::TRACE,
        "debug" => tracing::level_filters::LevelFilter::DEBUG,
        "warn" | "warning" => tracing::level_filters::LevelFilter::WARN,
        "error" => tracing::level_filters::LevelFilter::ERROR,
        _ => tracing::level_filters::LevelFilter::INFO,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load `.env` for secrets such as OPENAI_API_KEY; absence is fine.
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt().with_max_level(log_level()).init();

    tracing::info!("Starting DynamicLLM Runtime");

    let config_dir = RuntimeConfig::config_dir();
    let config = RuntimeConfig::load(&config_dir);
    let models = match ModelsFile::load(&config_dir) {
        Ok(Some(file)) => {
            tracing::info!("Loaded {} model(s) from config", file.models.len());
            file.models
        }
        Ok(None) => {
            tracing::warn!("No models.json found; using built-in model defaults");
            Vec::new()
        }
        Err(e) => {
            tracing::error!("{e}; using built-in model defaults");
            Vec::new()
        }
    };

    let runtime = Arc::new(core::runtime::Runtime::with_config(config.clone(), models).await?);
    let app = api::create_router(runtime);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("Listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for shutdown signal");
            tracing::info!("Shutdown signal received");
        })
        .await?;

    Ok(())
}
