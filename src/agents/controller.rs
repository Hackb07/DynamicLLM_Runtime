use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Lifecycle of one agent task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentState {
    Pending,
    Running,
    AwaitingTool,
    Done,
    Failed,
    Cancelled,
}

impl AgentState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            AgentState::Done | AgentState::Failed | AgentState::Cancelled
        )
    }
}

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_TRANSCRIPT_CAP: usize = 50;

/// Owns step budget, timeout, transcript, and cancellation for one task.
pub struct TaskController {
    max_steps: u32,
    timeout: Duration,
    state: AgentState,
    steps_taken: u32,
    started_at: Option<Instant>,
    cancel_flag: Arc<AtomicBool>,
    transcript: Vec<String>,
    transcript_cap: usize,
    error: Option<String>,
}

impl TaskController {
    pub fn new(max_steps: u32) -> Self {
        Self {
            max_steps,
            timeout: DEFAULT_TIMEOUT,
            state: AgentState::Pending,
            steps_taken: 0,
            started_at: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            transcript: Vec::new(),
            transcript_cap: DEFAULT_TRANSCRIPT_CAP,
            error: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_transcript_cap(mut self, cap: usize) -> Self {
        self.transcript_cap = cap;
        self.transcript_truncate();
        self
    }

    pub fn max_steps(&self) -> u32 {
        self.max_steps
    }

    pub fn state(&self) -> AgentState {
        self.state
    }

    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn transcript(&self) -> &[String] {
        &self.transcript
    }

    pub fn transcript_cap(&self) -> usize {
        self.transcript_cap
    }

    /// Shared cancellation flag; clones stay usable across tasks.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel_flag.clone()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Request cancellation. Moves a live task to `Cancelled`; a task
    /// that already finished keeps its terminal state.
    pub fn cancel(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        if !self.state.is_terminal() {
            self.state = AgentState::Cancelled;
        }
    }

    /// Move `Pending -> Running`. Fails if already started or cancelled.
    pub fn start(&mut self) -> Result<()> {
        if self.is_cancelled() {
            self.state = AgentState::Cancelled;
            return Err(Error::Cancelled);
        }
        if self.state != AgentState::Pending {
            return Err(Error::Validation("task already started".to_string()));
        }
        self.state = AgentState::Running;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    /// Fail when cancelled or past the deadline, recording the state.
    pub fn check_limits(&mut self) -> Result<()> {
        if self.is_cancelled() {
            if !self.state.is_terminal() {
                self.state = AgentState::Cancelled;
            }
            return Err(Error::Cancelled);
        }
        if let Some(started) = self.started_at {
            if started.elapsed() > self.timeout {
                self.state = AgentState::Failed;
                self.error = Some("timeout exceeded".to_string());
                return Err(Error::Timeout);
            }
        }
        Ok(())
    }

    /// Open one budgeted step. Returns the 1-based step number.
    /// Enforces started state, cancellation, timeout, and step budget.
    pub fn begin_step(&mut self) -> Result<u32> {
        if self.state == AgentState::Pending {
            return Err(Error::Validation("task not started".to_string()));
        }
        if self.state.is_terminal() {
            return Err(Error::Validation("task already finished".to_string()));
        }
        self.check_limits()?;
        if self.steps_taken >= self.max_steps {
            self.state = AgentState::Failed;
            self.error = Some("step budget exceeded".to_string());
            return Err(Error::Validation("step budget exceeded".to_string()));
        }
        self.steps_taken += 1;
        Ok(self.steps_taken)
    }

    pub fn mark_awaiting_tool(&mut self) {
        if matches!(self.state, AgentState::Running | AgentState::AwaitingTool) {
            self.state = AgentState::AwaitingTool;
        }
    }

    pub fn mark_running(&mut self) {
        if matches!(self.state, AgentState::Running | AgentState::AwaitingTool) {
            self.state = AgentState::Running;
        }
    }

    /// Append a transcript entry, dropping the oldest past the cap.
    pub fn push_transcript(&mut self, entry: String) {
        self.transcript.push(entry);
        self.transcript_truncate();
    }

    fn transcript_truncate(&mut self) {
        while self.transcript.len() > self.transcript_cap {
            self.transcript.remove(0);
        }
    }

    pub fn finish_success(&mut self) {
        if !self.state.is_terminal() {
            self.state = AgentState::Done;
        }
    }

    pub fn finish_failed(&mut self, reason: impl Into<String>) {
        if !self.state.is_terminal() {
            self.state = AgentState::Failed;
            self.error = Some(reason.into());
        }
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|started| started.elapsed())
    }

    pub fn remaining_time(&self) -> Duration {
        match self.started_at {
            Some(started) => self.timeout.saturating_sub(started.elapsed()),
            None => self.timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_pending_to_done() {
        let mut task = TaskController::new(3);
        assert_eq!(task.state(), AgentState::Pending);
        task.start().unwrap();
        assert_eq!(task.state(), AgentState::Running);
        assert_eq!(task.begin_step().unwrap(), 1);
        task.mark_awaiting_tool();
        assert_eq!(task.state(), AgentState::AwaitingTool);
        task.mark_running();
        assert_eq!(task.state(), AgentState::Running);
        task.finish_success();
        assert_eq!(task.state(), AgentState::Done);
        assert!(task.begin_step().is_err());
    }

    #[test]
    fn step_budget_enforcement_fails_the_task() {
        let mut task = TaskController::new(1);
        task.start().unwrap();
        assert_eq!(task.begin_step().unwrap(), 1);
        let err = task.begin_step().unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert_eq!(task.state(), AgentState::Failed);
        assert_eq!(task.error(), Some("step budget exceeded"));
    }

    #[test]
    fn requires_start_before_steps_and_rejects_double_start() {
        let mut task = TaskController::new(2);
        assert!(task.begin_step().is_err());
        task.start().unwrap();
        assert!(task.start().is_err());
    }

    #[test]
    fn transcript_drops_oldest_past_cap() {
        let mut task = TaskController::new(10).with_transcript_cap(3);
        for i in 0..5 {
            task.push_transcript(format!("entry-{i}"));
        }
        assert_eq!(
            task.transcript(),
            &[
                "entry-2".to_string(),
                "entry-3".to_string(),
                "entry-4".to_string()
            ]
        );
    }

    #[test]
    fn cancellation_wins_and_sticks() {
        let mut task = TaskController::new(5);
        task.start().unwrap();
        let handle = task.cancel_handle();
        assert!(!task.is_cancelled());
        handle.store(true, Ordering::SeqCst);
        let err = task.check_limits().unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        assert_eq!(task.state(), AgentState::Cancelled);

        // Pre-cancelled tasks refuse to start.
        let mut task = TaskController::new(5);
        task.cancel();
        assert!(matches!(task.start().unwrap_err(), Error::Cancelled));
        assert_eq!(task.state(), AgentState::Cancelled);
    }

    #[test]
    fn timeout_fails_the_task() {
        let mut task = TaskController::new(5).with_timeout(Duration::from_millis(5));
        task.start().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let err = task.begin_step().unwrap_err();
        assert!(matches!(err, Error::Timeout));
        assert_eq!(task.state(), AgentState::Failed);
    }

    #[test]
    fn terminal_states_are_sticky() {
        let mut task = TaskController::new(2);
        task.start().unwrap();
        task.finish_failed("boom");
        assert_eq!(task.state(), AgentState::Failed);
        task.finish_success();
        assert_eq!(task.state(), AgentState::Failed);
        task.cancel();
        assert_eq!(task.state(), AgentState::Failed);
    }
}
