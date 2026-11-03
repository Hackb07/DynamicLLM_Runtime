use crate::agents::types::{redact_secrets, rfc3339_utc, unix_now_secs, AuditEvent};
use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Component, PathBuf};

/// Maximum bytes returned by one `read_file` call.
pub const MAX_READ_BYTES: usize = 64 * 1024;
/// Maximum characters accepted by `echo`.
pub const MAX_ECHO_CHARS: usize = 32 * 1024;
/// Maximum expression length accepted by `calc`.
pub const MAX_EXPR_CHARS: usize = 1024;

pub const TOOL_NOW: &str = "now";
pub const TOOL_ECHO: &str = "echo";
pub const TOOL_CALC: &str = "calc";
pub const TOOL_READ_FILE: &str = "read_file";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Deny-by-default registry for the safe tool subset. An empty registry
/// executes nothing; execution additionally requires the tool to appear
/// in the caller's allowlist.
pub struct ToolRegistry {
    tools: Vec<ToolDefinition>,
    workspace_root: PathBuf,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            workspace_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    pub fn with_workspace_root(workspace_root: PathBuf) -> Self {
        Self {
            tools: Vec::new(),
            workspace_root,
        }
    }

    /// Registry pre-loaded with the safe subset: `now`, `echo`, `calc`,
    /// `read_file`. File reads are constrained to `workspace_root`.
    pub fn with_safe_subset(workspace_root: PathBuf) -> Self {
        let mut registry = Self::with_workspace_root(workspace_root);
        registry.register_safe_subset();
        registry
    }

    pub fn workspace_root(&self) -> &PathBuf {
        &self.workspace_root
    }

    pub fn register(&mut self, tool: ToolDefinition) {
        if let Some(existing) = self.tools.iter_mut().find(|t| t.name == tool.name) {
            *existing = tool;
        } else {
            self.tools.push(tool);
        }
    }

    /// Register (or refresh) the four safe-subset definitions.
    pub fn register_safe_subset(&mut self) {
        for tool in safe_tool_definitions() {
            self.register(tool);
        }
    }

    pub fn list(&self) -> &Vec<ToolDefinition> {
        &self.tools
    }

    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|tool| tool.name == name)
    }

    pub fn is_allowed(&self, name: &str, allowed_tools: &[String]) -> bool {
        self.get(name).is_some() && allowed_tools.iter().any(|a| a == name)
    }

    /// Tool definitions filtered to registered tools on the allowlist,
    /// converted to the core type used by inference requests.
    pub fn allowed_definitions(&self, allowed_tools: &[String]) -> Vec<crate::core::types::Tool> {
        self.tools
            .iter()
            .filter(|tool| allowed_tools.iter().any(|a| a == &tool.name))
            .map(|tool| crate::core::types::Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect()
    }

    /// Validate argument shape for a registered tool. Unknown tools fail.
    pub fn validate_args(&self, name: &str, args: &serde_json::Value) -> Result<()> {
        if self.get(name).is_none() {
            return Err(Error::Validation(format!("unknown tool: {name}")));
        }
        match name {
            TOOL_NOW => {
                if !(args.is_null() || args.is_object()) {
                    return Err(Error::Validation(
                        "now expects an object or no arguments".to_string(),
                    ));
                }
                Ok(())
            }
            TOOL_ECHO => {
                let text = args.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::Validation("echo requires a string field 'text'".to_string())
                })?;
                if text.chars().count() > MAX_ECHO_CHARS {
                    return Err(Error::Validation("echo text too long".to_string()));
                }
                Ok(())
            }
            TOOL_CALC => {
                let expr = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        Error::Validation("calc requires a string field 'expression'".to_string())
                    })?;
                if expr.trim().is_empty() {
                    return Err(Error::Validation(
                        "calc expression must not be empty".to_string(),
                    ));
                }
                if expr.chars().count() > MAX_EXPR_CHARS {
                    return Err(Error::Validation("calc expression too long".to_string()));
                }
                Ok(())
            }
            TOOL_READ_FILE => {
                let path = args.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::Validation("read_file requires a string field 'path'".to_string())
                })?;
                if path.trim().is_empty() {
                    return Err(Error::Validation(
                        "read_file path must not be empty".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(Error::Validation(format!(
                "tool '{name}' is registered but not executable in the safe subset"
            ))),
        }
    }

    /// Execute a tool and return its JSON result plus an audit record.
    /// Unknown tools, tools missing from `allowed_tools`, and invalid
    /// arguments all error without executing anything.
    pub fn execute_tool(
        &self,
        name: &str,
        args: serde_json::Value,
        allowed_tools: &[String],
    ) -> Result<(serde_json::Value, AuditEvent)> {
        if self.get(name).is_none() {
            return Err(Error::Validation(format!("unknown tool: {name}")));
        }
        if !allowed_tools.iter().any(|a| a == name) {
            return Err(Error::Forbidden(format!(
                "tool denied by allowlist: {name}"
            )));
        }
        self.validate_args(name, &args)?;

        let value = match name {
            TOOL_NOW => tool_now(),
            TOOL_ECHO => {
                let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                serde_json::json!({ "text": text })
            }
            TOOL_CALC => {
                let expr = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let result = eval_arithmetic(expr)?;
                let number = json_number(result);
                serde_json::json!({ "expression": expr, "result": number })
            }
            TOOL_READ_FILE => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_read_file(path)?
            }
            _ => {
                return Err(Error::Validation(format!(
                    "tool '{name}' is not executable in the safe subset"
                )));
            }
        };

        let event = AuditEvent::tool_call(0, name, redact_secrets(&args), true, None);
        Ok((value, event))
    }

    fn tool_read_file(&self, path: &str) -> Result<serde_json::Value> {
        let rel = std::path::Path::new(path);
        if rel.is_absolute() {
            return Err(Error::Validation(
                "read_file rejects absolute paths".to_string(),
            ));
        }
        for component in rel.components() {
            match component {
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(Error::Validation(
                        "read_file rejects escapes outside the workspace".to_string(),
                    ));
                }
                _ => {}
            }
        }

        let joined = self.workspace_root.join(rel);
        let canonical_root = self
            .workspace_root
            .canonicalize()
            .unwrap_or_else(|_| self.workspace_root.clone());
        let canonical = joined.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Validation(format!("file not found: {path}"))
            } else {
                Error::Io(e)
            }
        })?;
        if !canonical.starts_with(&canonical_root) {
            return Err(Error::Validation(
                "read_file rejects escapes outside the workspace".to_string(),
            ));
        }

        let file = std::fs::File::open(&canonical)?;
        let mut limited = file.take((MAX_READ_BYTES + 1) as u64);
        let mut bytes = Vec::new();
        limited.read_to_end(&mut bytes)?;
        let truncated = bytes.len() > MAX_READ_BYTES;
        bytes.truncate(MAX_READ_BYTES);
        let content = String::from_utf8_lossy(&bytes).into_owned();
        Ok(serde_json::json!({
            "path": path,
            "content": content,
            "bytes": bytes.len(),
            "truncated": truncated,
        }))
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn safe_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: TOOL_NOW.to_string(),
            description: "Current UTC time as unix seconds and RFC 3339.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }),
        },
        ToolDefinition {
            name: TOOL_ECHO.to_string(),
            description: "Echo back the given text unchanged.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }),
        },
        ToolDefinition {
            name: TOOL_CALC.to_string(),
            description: "Evaluate a basic arithmetic expression (+ - * / % ^ and parentheses)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "expression": { "type": "string" } },
                "required": ["expression"],
            }),
        },
        ToolDefinition {
            name: TOOL_READ_FILE.to_string(),
            description: "Read a UTF-8 workspace file by relative path (byte-capped).".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
        },
    ]
}

fn tool_now() -> serde_json::Value {
    let secs = unix_now_secs();
    serde_json::json!({ "unix_seconds": secs, "utc": rfc3339_utc(secs) })
}

fn json_number(value: f64) -> serde_json::Value {
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        serde_json::Value::Number(serde_json::Number::from(value as i64))
    } else {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}

/// Evaluate `+ - * / % ^`, parentheses, unary signs, and decimals.
/// Anything else (names, calls, shell) is rejected.
pub fn eval_arithmetic(input: &str) -> Result<f64> {
    let mut parser = CalcParser {
        chars: input.chars().collect(),
        pos: 0,
    };
    let value = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.chars.len() {
        return Err(Error::Validation(format!(
            "malformed expression near '{}'",
            parser.rest()
        )));
    }
    if !value.is_finite() {
        return Err(Error::Validation("non-finite result".to_string()));
    }
    Ok(value)
}

struct CalcParser {
    chars: Vec<char>,
    pos: usize,
}

impl CalcParser {
    fn rest(&self) -> String {
        self.chars[self.pos..].iter().collect()
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<f64> {
        let mut value = self.parse_term()?;
        loop {
            if self.eat('+') {
                value += self.parse_term()?;
            } else if self.eat('-') {
                value -= self.parse_term()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_term(&mut self) -> Result<f64> {
        let mut value = self.parse_factor()?;
        loop {
            if self.eat('*') {
                value *= self.parse_factor()?;
            } else if self.eat('/') {
                let rhs = self.parse_factor()?;
                if rhs == 0.0 {
                    return Err(Error::Validation("division by zero".to_string()));
                }
                value /= rhs;
            } else if self.eat('%') {
                let rhs = self.parse_factor()?;
                if rhs == 0.0 {
                    return Err(Error::Validation("division by zero".to_string()));
                }
                value %= rhs;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_factor(&mut self) -> Result<f64> {
        let base = self.parse_unary()?;
        if self.eat('^') {
            let exp = self.parse_factor()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<f64> {
        if self.eat('+') {
            return self.parse_unary();
        }
        if self.eat('-') {
            return Ok(-self.parse_unary()?);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<f64> {
        if self.eat('(') {
            let value = self.parse_expr()?;
            if !self.eat(')') {
                return Err(Error::Validation("missing closing ')'".to_string()));
            }
            return Ok(value);
        }
        self.parse_number()
    }

    fn parse_number(&mut self) -> Result<f64> {
        self.skip_ws();
        let start = self.pos;
        let mut seen_digit = false;
        let mut seen_dot = false;
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if c.is_ascii_digit() {
                seen_digit = true;
                self.pos += 1;
            } else if c == '.' && !seen_dot {
                seen_dot = true;
                self.pos += 1;
            } else {
                break;
            }
        }
        if !seen_digit {
            return Err(Error::Validation(format!(
                "expected a number near '{}'",
                self.rest()
            )));
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .parse::<f64>()
            .map_err(|_| Error::Validation("invalid number".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ToolRegistry {
        ToolRegistry::with_safe_subset(PathBuf::from("."))
    }

    fn allow(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn safe_subset_registers_four_tools() {
        let registry = registry();
        assert_eq!(registry.list().len(), 4);
        assert!(registry.get("now").is_some());
        assert!(registry.get("echo").is_some());
        assert!(registry.get("calc").is_some());
        assert!(registry.get("read_file").is_some());
    }

    #[test]
    fn unknown_tool_never_executes() {
        let registry = registry();
        let err = registry
            .execute_tool("shell", serde_json::json!({}), &allow(&["shell"]))
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(err.to_string().contains("unknown tool"));
    }

    #[test]
    fn allowlist_denies_by_default() {
        let registry = registry();
        // Registered but not allowlisted.
        let err = registry
            .execute_tool("echo", serde_json::json!({"text": "hi"}), &[])
            .unwrap_err();
        assert!(err.to_string().contains("denied"));
        // Empty registry denies everything.
        let empty = ToolRegistry::new();
        assert!(empty
            .execute_tool("echo", serde_json::json!({"text": "hi"}), &allow(&["echo"]))
            .is_err());
    }

    #[test]
    fn now_returns_utc_timestamp() {
        let registry = registry();
        let (value, event) = registry
            .execute_tool("now", serde_json::json!({}), &allow(&["now"]))
            .unwrap();
        assert!(value["unix_seconds"].as_u64().unwrap() > 1_700_000_000);
        let utc = value["utc"].as_str().unwrap();
        assert!(utc.ends_with('Z') && utc.contains('T'), "{utc}");
        assert!(event.success);
    }

    #[test]
    fn echo_roundtrips_and_validates_args() {
        let registry = registry();
        let allowed = allow(&["echo"]);
        let (value, _) = registry
            .execute_tool("echo", serde_json::json!({"text": "hi"}), &allowed)
            .unwrap();
        assert_eq!(value["text"], "hi");
        assert!(registry
            .execute_tool("echo", serde_json::json!({}), &allowed)
            .is_err());
        assert!(registry
            .execute_tool("echo", serde_json::json!({"text": 1}), &allowed)
            .is_err());
    }

    #[test]
    fn calc_cases() {
        let cases = [
            ("2+3*4", 14.0),
            ("(2+3)*4", 20.0),
            ("10/4", 2.5),
            ("10%3", 1.0),
            ("2^3", 8.0),
            ("-3 + +2", -1.0),
            ("  1.5 * 2 ", 3.0),
            ("2^3^2", 512.0),
            ("-(2+3)", -5.0),
        ];
        for (expr, expected) in cases {
            let got = eval_arithmetic(expr).unwrap();
            assert!((got - expected).abs() < 1e-9, "{expr} -> {got}");
        }
        let (value, _) = registry()
            .execute_tool(
                "calc",
                serde_json::json!({"expression": "2+3"}),
                &allow(&["calc"]),
            )
            .unwrap();
        assert_eq!(value["result"], 5);
    }

    #[test]
    fn calc_rejects_malformed_and_unsafe_input() {
        for bad in [
            "",
            "   ",
            "2+",
            "(2",
            "2)",
            "abc",
            "2+foo",
            "rm -rf /",
            "1;2",
            "`id`",
            "2**3",
            "1/0",
            "1%0",
            "999999999999999999999^999999",
        ] {
            assert!(eval_arithmetic(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn read_file_serves_workspace_files_and_caps_bytes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.txt"), "hello").unwrap();
        let big = "x".repeat(MAX_READ_BYTES + 100);
        std::fs::write(dir.path().join("big.txt"), &big).unwrap();

        let registry = ToolRegistry::with_safe_subset(dir.path().to_path_buf());
        let allowed = allow(&["read_file"]);
        let (value, _) = registry
            .execute_tool(
                "read_file",
                serde_json::json!({"path": "note.txt"}),
                &allowed,
            )
            .unwrap();
        assert_eq!(value["content"], "hello");
        assert_eq!(value["truncated"], false);

        let (value, _) = registry
            .execute_tool(
                "read_file",
                serde_json::json!({"path": "big.txt"}),
                &allowed,
            )
            .unwrap();
        assert_eq!(value["truncated"], true);
        assert_eq!(value["bytes"], MAX_READ_BYTES);
        assert_eq!(value["content"].as_str().unwrap().len(), MAX_READ_BYTES);
    }

    #[test]
    fn read_file_rejects_traversal_absolute_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.txt"), "ok").unwrap();
        let registry = ToolRegistry::with_safe_subset(dir.path().to_path_buf());
        let allowed = allow(&["read_file"]);

        for bad in ["../ok.txt", "sub/../../ok.txt", "..", "/etc/hostname"] {
            let err = registry
                .execute_tool("read_file", serde_json::json!({"path": bad}), &allowed)
                .unwrap_err();
            assert!(
                err.to_string().contains("absolute") || err.to_string().contains("escapes"),
                "{bad}: {err}"
            );
        }
        let err = registry
            .execute_tool(
                "read_file",
                serde_json::json!({"path": "missing.txt"}),
                &allowed,
            )
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }
}
