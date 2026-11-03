//! Basic in-process SDK usage: list models, read status, run inference,
//! and run one agentic task. Backend calls need a reachable model
//! (see QUICKSTART.md); without one, the infer/agent steps print their
//! error and the example still exits 0.
//!
//! Run with: `cargo run --example sdk_basic`

use dynamic_llm_runtime::sdk::{AgentTaskBuilder, Client, InferBuilder};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Same files the server reads; missing files fall back to defaults.
    let client = Client::from_config_dir("config").await?;

    let models = client.list_models().await;
    println!("registered models: {}", models.len());
    for model in &models {
        println!("  - {} (backend: {})", model.id, model.backend);
    }

    let status = client.status().await;
    println!(
        "status: {}, telemetry total requests: {}",
        status["status"], status["telemetry"]["total_requests"]
    );

    // Non-streaming inference (pinned model avoids router surprises).
    let request = InferBuilder::new("Say hi in one sentence.")
        .max_tokens(32)
        .build();
    match client.infer(request).await {
        Ok(response) => println!("infer [{}]: {}", response.model, response.content),
        Err(e) => println!("infer skipped (no reachable backend?): {e}"),
    }

    // Streaming inference: print chunks as they arrive.
    let stream_request = InferBuilder::new("Count to three.")
        .stream(true)
        .max_tokens(32)
        .build();
    match client.infer_stream(stream_request).await {
        Ok(mut chunks) => {
            print!("stream: ");
            while let Some(chunk) = chunks.next().await {
                match chunk {
                    Ok(text) => print!("{text}"),
                    Err(e) => {
                        println!("\nstream chunk error: {e}");
                        break;
                    }
                }
            }
            println!();
        }
        Err(e) => println!("stream skipped (no reachable backend?): {e}"),
    }

    // Agentic task with the `calc` tool allowlisted (deny-by-default:
    // only listed tools may execute).
    let task = AgentTaskBuilder::new("Compute 2+3")
        .allow_tool("calc")
        .max_steps(4)
        .timeout_secs(30)
        .build();
    match client.run_agent(task).await {
        Ok(response) if response.is_success() => {
            println!("agent ({} steps): {}", response.steps_used, response.answer)
        }
        Ok(response) => println!(
            "agent run ended as {:?}: {}",
            response.state,
            response.error.as_deref().unwrap_or("unknown")
        ),
        Err(e) => println!("agent rejected: {e}"),
    }

    Ok(())
}
