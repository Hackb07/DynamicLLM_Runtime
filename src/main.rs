use dynamic_llm_runtime::{api, core, telemetry};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dynamic_llm_runtime=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting DynamicLLM Runtime");

    let app = api::create_router();
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    tracing::info!("Listening on http://127.0.0.1:8080");
    axum::serve(listener, app).await?;

    Ok(())
}
