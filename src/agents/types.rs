use crate::agents::controller::AgentState;
use serde::{Deserialize, Serialize};

/// Agent-level request: a task prompt plus the limits and tool allowlist
/// the run must obey. The allowlist is deny-by-default: an empty list
/// means no tool may execute.
fn default_max_steps() -> u32 {
    8
}

fn default_timeout_secs() -> u64 {
    60
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    /// Task text. Accepts `task` as an alias for HTTP clients.
    #[serde(alias = "task")]
    pub prompt: String,
    #[serde(default, alias = "tools")]
    pub allowed_tools: Vec<String>,
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
    #[serde(default = "default_timeout_secs", alias = "timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub token_budget: Option<u32>,
    #[serde(default, alias = "model_hint")]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens_per_step: Option<u32>,
}

impl AgentRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            allowed_tools: Vec::new(),
            max_steps: 8,
            timeout_secs: 60,
            token_budget: None,
            model: None,
            max_tokens_per_step: None,
        }
    }

    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_token_budget(mut self, budget: u32) -> Self {
        self.token_budget = Some(budget);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Outcome of an agent run. Terminal failures are reported as a response
/// with `state` set to `Failed`/`Cancelled` and `error` populated (so the
/// audit trail survives); only malformed requests return `Err`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub answer: String,
    pub steps_used: u32,
    pub state: AgentState,
    pub tokens_used: u32,
    pub audit: Vec<AuditEvent>,
    /// Redacted per-step transcript (user/assistant/tool lines, capped).
    #[serde(default)]
    pub transcript: Vec<String>,
    pub error: Option<String>,
}

impl AgentResponse {
    pub fn is_success(&self) -> bool {
        self.state == AgentState::Done && self.error.is_none()
    }
}

/// One redacted audit record for an inference step or tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub step: u32,
    /// "inference", "tool_call", "done", "failed", or "cancelled".
    pub kind: String,
    pub tool: Option<String>,
    /// Tool arguments with secrets redacted, when applicable.
    pub arguments: Option<serde_json::Value>,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp_unix_secs: u64,
}

impl AuditEvent {
    pub fn inference(step: u32, success: bool, error: Option<String>) -> Self {
        Self {
            step,
            kind: "inference".to_string(),
            tool: None,
            arguments: None,
            success,
            error,
            timestamp_unix_secs: unix_now_secs(),
        }
    }

    pub fn tool_call(
        step: u32,
        tool: &str,
        redacted_args: serde_json::Value,
        success: bool,
        error: Option<String>,
    ) -> Self {
        Self {
            step,
            kind: "tool_call".to_string(),
            tool: Some(tool.to_string()),
            arguments: Some(redacted_args),
            success,
            error,
            timestamp_unix_secs: unix_now_secs(),
        }
    }

    pub fn terminal(step: u32, kind: &str, success: bool, error: Option<String>) -> Self {
        Self {
            step,
            kind: kind.to_string(),
            tool: None,
            arguments: None,
            success,
            error,
            timestamp_unix_secs: unix_now_secs(),
        }
    }
}

/// Seconds since the Unix epoch (UTC), without extra dependencies.
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// RFC 3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) from epoch seconds.
pub fn rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year,
        month,
        day,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's days-to-civil conversion for non-negative day counts.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

const SECRET_SUBSTRINGS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "bearer",
    "private",
    "credential",
    "cookie",
    "session",
    "access_key",
    "secret_key",
];

fn is_secret_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    if lower == "auth" || lower == "key" {
        return true;
    }
    SECRET_SUBSTRINGS.iter().any(|hint| lower.contains(hint))
}

fn looks_like_secret_value(value: &str) -> bool {
    let trimmed = value.trim();
    let lower_prefix: String = trimmed.chars().take(7).collect::<String>().to_lowercase();
    lower_prefix.starts_with("bearer ") || trimmed.starts_with("sk-") || trimmed.starts_with("sk_")
}

/// Deep-copy a JSON value with secrets replaced by `"[REDACTED]"`.
/// Object values under secret-like keys are redacted, as are string
/// values that look like bearer tokens or API keys.
pub fn redact_secrets(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut redacted = serde_json::Map::with_capacity(map.len());
            for (key, val) in map {
                if is_secret_key(key) {
                    redacted.insert(
                        key.clone(),
                        serde_json::Value::String("[REDACTED]".to_string()),
                    );
                } else {
                    redacted.insert(key.clone(), redact_secrets(val));
                }
            }
            serde_json::Value::Object(redacted)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(redact_secrets).collect())
        }
        serde_json::Value::String(text) => {
            if looks_like_secret_value(text) {
                serde_json::Value::String("[REDACTED]".to_string())
            } else {
                value.clone()
            }
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_secret_keys_recursively() {
        let input = serde_json::json!({
            "text": "hello",
            "api_key": "sk-abc",
            "nested": { "Password": "hunter2", "ok": 1 },
            "list": [{ "token": "t", "n": 2 }]
        });
        let out = redact_secrets(&input);
        assert_eq!(out["text"], "hello");
        assert_eq!(out["api_key"], "[REDACTED]");
        assert_eq!(out["nested"]["Password"], "[REDACTED]");
        assert_eq!(out["nested"]["ok"], 1);
        assert_eq!(out["list"][0]["token"], "[REDACTED]");
        assert_eq!(out["list"][0]["n"], 2);
    }

    #[test]
    fn redacts_bearer_like_values_but_keeps_plain_text() {
        assert_eq!(
            redact_secrets(&serde_json::json!("Bearer abc123")),
            serde_json::json!("[REDACTED]")
        );
        assert_eq!(
            redact_secrets(&serde_json::json!("just words")),
            serde_json::json!("just words")
        );
        // "author" must not be treated as an auth secret.
        let out = redact_secrets(&serde_json::json!({ "author": "ada" }));
        assert_eq!(out["author"], "ada");
    }

    #[test]
    fn rfc3339_formats_epoch_and_known_date() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        // 2024-01-01T00:00:00Z
        assert_eq!(rfc3339_utc(1_704_067_200), "2024-01-01T00:00:00Z");
        let now = unix_now_secs();
        assert!(now > 1_700_000_000);
    }
}
