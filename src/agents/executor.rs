use crate::agents::controller::{AgentState, TaskController};
use crate::agents::tools::ToolRegistry;
use crate::agents::types::{redact_secrets, AgentRequest, AgentResponse, AuditEvent};
use crate::core::error::{Error, Result};
use crate::core::runtime::Runtime;
use crate::core::types::{InferenceRequest, InferenceResponse};
use std::path::PathBuf;

/// Runs the prompt -> infer -> tool -> feedback loop against a runtime
/// (or any injected async inference function, for tests).
pub struct AgentExecutor {
    registry: ToolRegistry,
}

impl AgentExecutor {
    pub fn new() -> Self {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            registry: ToolRegistry::with_safe_subset(root),
        }
    }

    pub fn with_registry(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub fn with_workspace_root(workspace_root: PathBuf) -> Self {
        Self {
            registry: ToolRegistry::with_safe_subset(workspace_root),
        }
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Run an agent task against the real runtime.
    pub async fn run(&self, runtime: &Runtime, request: AgentRequest) -> Result<AgentResponse> {
        self.run_with_infer(request, |req| runtime.infer(req)).await
    }

    /// Run with an injected inference function (no network needed).
    pub async fn run_with_infer<F, Fut>(
        &self,
        request: AgentRequest,
        infer: F,
    ) -> Result<AgentResponse>
    where
        F: FnMut(InferenceRequest) -> Fut,
        Fut: std::future::Future<Output = Result<InferenceResponse>>,
    {
        validate_request(&request)?;
        let mut controller = TaskController::new(request.max_steps)
            .with_timeout(std::time::Duration::from_secs(request.timeout_secs));
        self.run_with_controller(request, &mut controller, infer)
            .await
    }

    /// Run with a caller-owned controller (exposes cancellation control).
    pub async fn run_with_controller<F, Fut>(
        &self,
        request: AgentRequest,
        controller: &mut TaskController,
        mut infer: F,
    ) -> Result<AgentResponse>
    where
        F: FnMut(InferenceRequest) -> Fut,
        Fut: std::future::Future<Output = Result<InferenceResponse>>,
    {
        validate_request(&request)?;
        if controller.state() == AgentState::Pending {
            if let Err(e) = controller.start() {
                return Ok(cancelled_or_failed(&request, controller, e));
            }
        }
        if controller.state() == AgentState::Cancelled {
            return Ok(cancelled_response(controller, "cancelled"));
        }

        let tool_defs = self.registry.allowed_definitions(&request.allowed_tools);
        let mut working_prompt = request.prompt.clone();
        let mut tokens_used: u32 = 0;
        let mut audit: Vec<AuditEvent> = Vec::new();
        controller.push_transcript(format!("user: {}", redact_text(&request.prompt)));

        loop {
            let step = match controller.begin_step() {
                Ok(step) => step,
                Err(e) => {
                    let (state, msg) = match &e {
                        Error::Cancelled => (AgentState::Cancelled, "cancelled".to_string()),
                        Error::Timeout => (AgentState::Failed, "timeout exceeded".to_string()),
                        _ => (
                            AgentState::Failed,
                            controller
                                .error()
                                .unwrap_or("step budget exceeded")
                                .to_string(),
                        ),
                    };
                    return Ok(AgentResponse {
                        answer: String::new(),
                        steps_used: controller.steps_taken(),
                        state,
                        tokens_used,
                        audit,
                        transcript: controller.transcript().to_vec(),
                        error: Some(msg),
                    });
                }
            };

            let infer_request = InferenceRequest {
                request_id: None,
                prompt: working_prompt.clone(),
                model: request.model.clone(),
                max_tokens: request.max_tokens_per_step,
                temperature: None,
                top_p: None,
                stop: None,
                stream: None,
                tools: if tool_defs.is_empty() {
                    None
                } else {
                    Some(tool_defs.clone())
                },
                metadata: None,
            };

            let remaining = controller.remaining_time();
            let response = match tokio::time::timeout(remaining, infer(infer_request)).await {
                Err(_) => {
                    controller.finish_failed("timeout exceeded");
                    audit.push(AuditEvent::terminal(
                        step,
                        "failed",
                        false,
                        Some("timeout exceeded".to_string()),
                    ));
                    return Ok(failed_response(
                        controller,
                        tokens_used,
                        audit,
                        "timeout exceeded",
                    ));
                }
                Ok(Err(e)) => {
                    controller.finish_failed(e.to_string());
                    audit.push(AuditEvent::inference(step, false, Some(e.to_string())));
                    let msg = e.to_string();
                    return Ok(failed_response(controller, tokens_used, audit, &msg));
                }
                Ok(Ok(response)) => response,
            };

            tokens_used += response_tokens(&response);
            if let Some(budget) = request.token_budget {
                if tokens_used > budget {
                    controller.finish_failed("token budget exceeded");
                    audit.push(AuditEvent::terminal(
                        step,
                        "failed",
                        false,
                        Some("token budget exceeded".to_string()),
                    ));
                    return Ok(failed_response(
                        controller,
                        tokens_used,
                        audit,
                        "token budget exceeded",
                    ));
                }
            }
            audit.push(AuditEvent::inference(step, true, None));
            controller.push_transcript(format!("assistant: {}", redact_text(&response.content)));

            let calls = response.tool_calls.clone().unwrap_or_default();
            if calls.is_empty() {
                if response.content.trim().is_empty() {
                    controller.finish_failed("empty final content");
                    audit.push(AuditEvent::terminal(
                        step,
                        "failed",
                        false,
                        Some("empty final content".to_string()),
                    ));
                    return Ok(failed_response(
                        controller,
                        tokens_used,
                        audit,
                        "empty final content",
                    ));
                }
                controller.finish_success();
                audit.push(AuditEvent::terminal(step, "done", true, None));
                return Ok(AgentResponse {
                    answer: response.content,
                    steps_used: controller.steps_taken(),
                    state: AgentState::Done,
                    tokens_used,
                    audit,
                    transcript: controller.transcript().to_vec(),
                    error: None,
                });
            }

            controller.mark_awaiting_tool();
            for call in &calls {
                let args = call.arguments.clone();
                match self
                    .registry
                    .execute_tool(&call.tool, args, &request.allowed_tools)
                {
                    Ok((value, mut event)) => {
                        event.step = step;
                        audit.push(event);
                        let summary = redact_text(&summarize_json(&value));
                        controller.push_transcript(format!("tool {} result: {summary}", call.tool));
                        working_prompt.push_str(&format!(
                            "\n[tool {} result]\n{}\n",
                            call.tool,
                            summarize_json(&redact_secrets(&value))
                        ));
                    }
                    Err(error) => {
                        let message = error.to_string();
                        controller.finish_failed(&message);
                        audit.push(AuditEvent::tool_call(
                            step,
                            &call.tool,
                            redact_secrets(&call.arguments),
                            false,
                            Some(message.clone()),
                        ));
                        return Ok(failed_response(controller, tokens_used, audit, &message));
                    }
                }
            }
            controller.mark_running();
        }
    }
}

fn validate_request(request: &AgentRequest) -> Result<()> {
    if request.prompt.trim().is_empty() {
        return Err(Error::Validation("prompt must not be empty".to_string()));
    }
    if request.max_steps == 0 {
        return Err(Error::Validation("max_steps must be positive".to_string()));
    }
    if request.timeout_secs == 0 {
        return Err(Error::Validation(
            "timeout_secs must be positive".to_string(),
        ));
    }
    if request.token_budget == Some(0) {
        return Err(Error::Validation(
            "token_budget must be positive".to_string(),
        ));
    }
    if request.max_tokens_per_step == Some(0) {
        return Err(Error::Validation(
            "max_tokens_per_step must be positive".to_string(),
        ));
    }
    Ok(())
}

fn response_tokens(response: &InferenceResponse) -> u32 {
    if let Some(usage) = &response.usage {
        usage.total_tokens
    } else {
        (response.content.len() as u32) / 4
    }
}

fn summarize_json(value: &serde_json::Value) -> String {
    match serde_json::to_string(value) {
        Ok(text) if text.len() <= 4000 => text,
        Ok(text) => format!("{}…[truncated]", &text[..4000]),
        Err(_) => "[unserializable]".to_string(),
    }
}

fn redact_text(text: &str) -> String {
    // Transcript strings are free text; mask inline bearer-looking tokens.
    let mut out = text.to_string();
    for prefix in ["Bearer ", "bearer "] {
        while let Some(pos) = out.find(prefix) {
            let end = out[pos + prefix.len()..]
                .find(|c: char| c.is_whitespace())
                .map(|i| pos + prefix.len() + i)
                .unwrap_or(out.len());
            out.replace_range(pos + prefix.len()..end, "[REDACTED]");
        }
    }
    out
}

fn failed_response(
    controller: &TaskController,
    tokens_used: u32,
    audit: Vec<AuditEvent>,
    message: &str,
) -> AgentResponse {
    AgentResponse {
        answer: String::new(),
        steps_used: controller.steps_taken(),
        state: controller.state(),
        tokens_used,
        audit,
        transcript: controller.transcript().to_vec(),
        error: Some(message.to_string()),
    }
}

fn cancelled_response(controller: &TaskController, message: &str) -> AgentResponse {
    AgentResponse {
        answer: String::new(),
        steps_used: controller.steps_taken(),
        state: AgentState::Cancelled,
        tokens_used: 0,
        audit: Vec::new(),
        transcript: controller.transcript().to_vec(),
        error: Some(message.to_string()),
    }
}

fn cancelled_or_failed(
    _request: &AgentRequest,
    controller: &TaskController,
    error: Error,
) -> AgentResponse {
    match error {
        Error::Cancelled => cancelled_response(controller, "cancelled"),
        other => AgentResponse {
            answer: String::new(),
            steps_used: controller.steps_taken(),
            state: controller.state(),
            tokens_used: 0,
            audit: Vec::new(),
            transcript: controller.transcript().to_vec(),
            error: Some(other.to_string()),
        },
    }
}

impl Default for AgentExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::ToolCall;

    fn final_response(content: &str) -> InferenceResponse {
        InferenceResponse {
            request_id: "r".to_string(),
            model: "mock".to_string(),
            content: content.to_string(),
            finish_reason: Some("stop".to_string()),
            usage: None,
            tool_calls: None,
            metadata: None,
        }
    }

    fn tool_response(tool: &str, args: serde_json::Value) -> InferenceResponse {
        InferenceResponse {
            request_id: "r".to_string(),
            model: "mock".to_string(),
            content: String::new(),
            finish_reason: Some("tool_calls".to_string()),
            usage: None,
            tool_calls: Some(vec![ToolCall {
                tool: tool.to_string(),
                arguments: args,
            }]),
            metadata: None,
        }
    }

    fn executor() -> AgentExecutor {
        AgentExecutor::with_workspace_root(PathBuf::from("."))
    }

    fn request(prompt: &str, tools: &[&str]) -> AgentRequest {
        AgentRequest::new(prompt).with_allowed_tools(tools.iter().map(|t| t.to_string()).collect())
    }

    #[tokio::test]
    async fn empty_prompt_is_rejected_upfront() {
        let err = executor()
            .run_with_infer(request("   ", &[]), |_| async { Ok(final_response("x")) })
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[tokio::test]
    async fn direct_answer_needs_no_tools() {
        let response = executor()
            .run_with_infer(request("hello", &[]), |req| async move {
                assert!(req.tools.is_none());
                Ok(final_response("hi there"))
            })
            .await
            .unwrap();
        assert!(response.is_success());
        assert_eq!(response.answer, "hi there");
        assert_eq!(response.steps_used, 1);
    }

    #[tokio::test]
    async fn tool_loop_feeds_results_back() {
        let response = executor()
            .run_with_infer(request("what is 2+3?", &["calc"]), |req| async move {
                if req.prompt.contains("[tool calc result]") {
                    Ok(final_response("The answer is 5"))
                } else {
                    assert!(req.tools.is_some());
                    Ok(tool_response(
                        "calc",
                        serde_json::json!({"expression": "2+3"}),
                    ))
                }
            })
            .await
            .unwrap();
        assert!(response.is_success());
        assert_eq!(response.answer, "The answer is 5");
        assert_eq!(response.steps_used, 2);
        let tool_events: Vec<_> = response
            .audit
            .iter()
            .filter(|e| e.kind == "tool_call")
            .collect();
        assert_eq!(tool_events.len(), 1);
        assert_eq!(tool_events[0].tool.as_deref(), Some("calc"));
        assert!(tool_events[0].success);
    }

    #[tokio::test]
    async fn disallowed_tool_fails_without_executing() {
        let response = executor()
            .run_with_infer(request("do it", &["echo"]), |_| async {
                Ok(tool_response(
                    "calc",
                    serde_json::json!({"expression": "1+1"}),
                ))
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("denied"));
        assert!(response.answer.is_empty());
    }

    #[tokio::test]
    async fn unknown_tool_fails_without_executing() {
        let response = executor()
            .run_with_infer(request("do it", &["shell"]), |_| async {
                Ok(tool_response("shell", serde_json::json!({"cmd": "id"})))
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("unknown tool"));
    }

    #[tokio::test]
    async fn step_budget_stops_chatty_models() {
        let req = request("loop", &["now"]).with_max_steps(2);
        let response = executor()
            .run_with_infer(req, |_| async {
                Ok(tool_response("now", serde_json::json!({})))
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert_eq!(response.steps_used, 2);
        assert!(response.error.as_deref().unwrap().contains("budget"));
    }

    #[tokio::test]
    async fn token_budget_is_enforced() {
        let req = request("hi", &[]).with_token_budget(5);
        let response = executor()
            .run_with_infer(req, |_| async {
                let mut resp = final_response("a long answer that costs tokens");
                resp.usage = Some(crate::core::types::Usage {
                    prompt_tokens: 3,
                    completion_tokens: 10,
                    total_tokens: 13,
                });
                Ok(resp)
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("token budget"));
    }

    #[tokio::test]
    async fn empty_final_content_fails() {
        let response = executor()
            .run_with_infer(request("hi", &[]), |_| async { Ok(final_response("   ")) })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn infer_errors_become_failed_responses_with_audit() {
        let response = executor()
            .run_with_infer(request("hi", &[]), |_| async {
                Err::<InferenceResponse, Error>(Error::Backend("down".to_string()))
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("down"));
        assert_eq!(response.audit.len(), 1);
        assert!(!response.audit[0].success);
    }

    #[tokio::test]
    async fn slow_inference_hits_the_timeout() {
        let req = request("hi", &[]).with_timeout_secs(1);
        // Shrink below one second via controller path.
        let mut controller =
            TaskController::new(4).with_timeout(std::time::Duration::from_millis(30));
        let response = executor()
            .run_with_controller(req, &mut controller, |_| async {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                Ok(final_response("too late"))
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Failed);
        assert!(response.error.as_deref().unwrap().contains("timeout"));
    }

    #[tokio::test]
    async fn pre_cancelled_controller_short_circuits() {
        let mut controller = TaskController::new(4);
        controller.cancel();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let moved = calls.clone();
        let response = executor()
            .run_with_controller(request("hi", &[]), &mut controller, move |_| {
                moved.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok(final_response("never")) }
            })
            .await
            .unwrap();
        assert_eq!(response.state, AgentState::Cancelled);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn audit_args_redact_secrets() {
        let response = executor()
            .run_with_infer(request("go", &["echo"]), |req| async move {
                if req.prompt.contains("[tool echo result]") {
                    Ok(final_response("done"))
                } else {
                    Ok(tool_response(
                        "echo",
                        serde_json::json!({"text": "hi", "api_key": "sk-live"}),
                    ))
                }
            })
            .await
            .unwrap();
        assert!(response.is_success());
        let tool_event = response
            .audit
            .iter()
            .find(|e| e.kind == "tool_call")
            .unwrap();
        let args = tool_event.arguments.as_ref().unwrap();
        assert_eq!(args["api_key"], "[REDACTED]");
        assert_eq!(args["text"], "hi");
    }
}
