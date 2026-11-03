use crate::agents::controller::AgentState;
use crate::core::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

/// Per-agent concurrency used when the caller does not specify one.
pub const DEFAULT_PER_AGENT_CONCURRENCY: usize = 4;
/// Transcript lines retained per agent (matches `TaskController`'s cap).
const SESSION_TRANSCRIPT_CAP: usize = 50;
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Independent task state + transcript for one agent inside a session.
#[derive(Debug, Clone)]
pub struct AgentTask {
    pub agent_id: String,
    pub state: AgentState,
    pub transcript: Vec<String>,
    pub steps_used: u32,
}

/// One isolated multi-agent session. Agents share the session's tool
/// policy but nothing else: every agent read/write requires the session
/// id, so there is no cross-session read path.
struct Session {
    allowed_tools: Vec<String>,
    agents: HashMap<String, AgentTask>,
    gates: HashMap<String, Arc<Semaphore>>,
    /// Last write activity; drives idle expiry in
    /// [`SessionManager::remove_idle_sessions`].
    last_active: std::time::Instant,
}

/// Holds one global permit plus one per-agent permit for a task.
/// Dropping the guard releases both.
#[derive(Debug)]
pub struct SessionPermit {
    _global: OwnedSemaphorePermit,
    _agent: OwnedSemaphorePermit,
}

/// In-memory manager for multi-agent sessions. Sessions are identified
/// by generated ids; each agent inside a session keeps independent task
/// state and transcript. Concurrency is bounded globally (across all
/// sessions/agents) and per agent.
pub struct SessionManager {
    sessions: Mutex<HashMap<String, Session>>,
    global: Arc<Semaphore>,
    global_bound: usize,
    per_agent_concurrency: usize,
    wait_timeout: Duration,
}

impl SessionManager {
    pub fn new(global_concurrency: usize, per_agent_concurrency: usize) -> Self {
        let global_bound = global_concurrency.max(1);
        Self {
            sessions: Mutex::new(HashMap::new()),
            global: Arc::new(Semaphore::new(global_bound)),
            global_bound,
            per_agent_concurrency: per_agent_concurrency.max(1),
            wait_timeout: DEFAULT_WAIT_TIMEOUT,
        }
    }

    pub fn with_wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    pub fn global_bound(&self) -> usize {
        self.global_bound
    }

    pub fn per_agent_concurrency(&self) -> usize {
        self.per_agent_concurrency
    }

    pub fn available_permits(&self) -> usize {
        self.global.available_permits()
    }

    /// Create an empty session with a deny-all tool policy.
    pub async fn create_session(&self) -> String {
        self.create_session_with_policy(Vec::new()).await
    }

    /// Create an empty session whose agents share `allowed_tools`.
    pub async fn create_session_with_policy(&self, allowed_tools: Vec<String>) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        self.sessions.lock().await.insert(
            id.clone(),
            Session {
                allowed_tools,
                agents: HashMap::new(),
                gates: HashMap::new(),
                last_active: std::time::Instant::now(),
            },
        );
        id
    }

    /// Drop a session and all of its agent state. Returns false when the
    /// session did not exist.
    pub async fn remove_session(&self, session_id: &str) -> bool {
        self.sessions.lock().await.remove(session_id).is_some()
    }

    /// Drop every session idle longer than `max_idle` (no writes since).
    /// Returns the number of sessions removed. Callers that create many
    /// short-lived sessions should call this periodically; long-lived
    /// servers can spawn a janitor task around it.
    pub async fn remove_idle_sessions(&self, max_idle: Duration) -> usize {
        let now = std::time::Instant::now();
        let mut sessions = self.sessions.lock().await;
        let before = sessions.len();
        sessions.retain(|_, session| now.duration_since(session.last_active) <= max_idle);
        before - sessions.len()
    }

    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Register an agent inside a session (idempotent). The agent starts
    /// `Pending` with an empty transcript.
    pub async fn spawn_agent(&self, session_id: &str, agent_id: &str) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        if !session.agents.contains_key(agent_id) {
            session.agents.insert(
                agent_id.to_string(),
                AgentTask {
                    agent_id: agent_id.to_string(),
                    state: AgentState::Pending,
                    transcript: Vec::new(),
                    steps_used: 0,
                },
            );
            session.gates.insert(
                agent_id.to_string(),
                Arc::new(Semaphore::new(self.per_agent_concurrency)),
            );
        }
        session.last_active = std::time::Instant::now();
        Ok(())
    }

    /// Read one agent's task state. Requires both ids: an agent in another
    /// session is unreadable (and indistinguishable from missing).
    pub async fn agent_task(&self, session_id: &str, agent_id: &str) -> Result<AgentTask> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        session
            .agents
            .get(agent_id)
            .cloned()
            .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))
    }

    pub async fn set_agent_state(
        &self,
        session_id: &str,
        agent_id: &str,
        state: AgentState,
    ) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        let agent = session
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))?;
        agent.state = state;
        session.last_active = std::time::Instant::now();
        Ok(())
    }

    pub async fn record_agent_steps(
        &self,
        session_id: &str,
        agent_id: &str,
        steps: u32,
    ) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        let agent = session
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))?;
        agent.steps_used = steps;
        session.last_active = std::time::Instant::now();
        Ok(())
    }

    /// Append one transcript line, dropping the oldest past the cap.
    pub async fn push_agent_transcript(
        &self,
        session_id: &str,
        agent_id: &str,
        entry: String,
    ) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        let agent = session
            .agents
            .get_mut(agent_id)
            .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))?;
        agent.transcript.push(entry);
        while agent.transcript.len() > SESSION_TRANSCRIPT_CAP {
            agent.transcript.remove(0);
        }
        session.last_active = std::time::Instant::now();
        Ok(())
    }

    /// The session's shared tool policy (cloned).
    pub async fn allowed_tools(&self, session_id: &str) -> Result<Vec<String>> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        Ok(session.allowed_tools.clone())
    }

    /// Whether `tool` is permitted by the session's shared policy.
    pub async fn is_tool_allowed(&self, session_id: &str, tool: &str) -> Result<bool> {
        Ok(self
            .allowed_tools(session_id)
            .await?
            .iter()
            .any(|allowed| allowed == tool))
    }

    /// A run's effective allowlist: requested tools intersected with the
    /// session's shared policy (deny-by-default on both sides).
    pub async fn effective_tools(
        &self,
        session_id: &str,
        requested: &[String],
    ) -> Result<Vec<String>> {
        let policy = self.allowed_tools(session_id).await?;
        Ok(requested
            .iter()
            .filter(|tool| policy.iter().any(|allowed| allowed == *tool))
            .cloned()
            .collect())
    }

    pub async fn agent_available_permits(&self, session_id: &str, agent_id: &str) -> Result<usize> {
        let sessions = self.sessions.lock().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
        let gate = session
            .gates
            .get(agent_id)
            .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))?;
        Ok(gate.available_permits())
    }

    /// Acquire one task slot for an agent: the global permit first, then
    /// the per-agent permit (fixed order, so holders never deadlock).
    /// Fails fast with `ResourceUnavailable` when either bound is
    /// exhausted past the wait timeout.
    pub async fn acquire(&self, session_id: &str, agent_id: &str) -> Result<SessionPermit> {
        let gate = {
            let sessions = self.sessions.lock().await;
            let session = sessions
                .get(session_id)
                .ok_or_else(|| Error::Validation(format!("unknown session: {session_id}")))?;
            session
                .gates
                .get(agent_id)
                .cloned()
                .ok_or_else(|| Error::Validation(format!("unknown agent: {agent_id}")))?
        };
        let global = acquire_with_timeout(&self.global, self.wait_timeout).await?;
        let agent = acquire_with_timeout(&gate, self.wait_timeout).await?;
        Ok(SessionPermit {
            _global: global,
            _agent: agent,
        })
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(16, DEFAULT_PER_AGENT_CONCURRENCY)
    }
}

async fn acquire_with_timeout(
    semaphore: &Arc<Semaphore>,
    wait_timeout: Duration,
) -> Result<OwnedSemaphorePermit> {
    match semaphore.clone().try_acquire_owned() {
        Ok(permit) => Ok(permit),
        Err(_) => tokio::time::timeout(wait_timeout, semaphore.clone().acquire_owned())
            .await
            .map_err(|_| {
                Error::ResourceUnavailable(
                    "agent concurrency limit reached; server is busy".to_string(),
                )
            })?
            .map_err(|_| Error::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager(global: usize, per_agent: usize) -> SessionManager {
        SessionManager::new(global, per_agent).with_wait_timeout(Duration::from_millis(50))
    }

    #[tokio::test]
    async fn sessions_isolate_agent_state() {
        let manager = test_manager(16, 4);
        let first = manager
            .create_session_with_policy(vec!["calc".to_string()])
            .await;
        let second = manager.create_session().await;

        // Same agent id in both sessions: fully independent state.
        manager.spawn_agent(&first, "worker").await.unwrap();
        manager.spawn_agent(&second, "worker").await.unwrap();
        manager
            .push_agent_transcript(&first, "worker", "first-only".to_string())
            .await
            .unwrap();
        manager
            .set_agent_state(&first, "worker", AgentState::Running)
            .await
            .unwrap();

        let task = manager.agent_task(&first, "worker").await.unwrap();
        assert_eq!(task.transcript, vec!["first-only".to_string()]);
        assert_eq!(task.state, AgentState::Running);

        let task = manager.agent_task(&second, "worker").await.unwrap();
        assert!(task.transcript.is_empty());
        assert_eq!(task.state, AgentState::Pending);

        // No cross-read: an agent visible in one session is unknown elsewhere.
        manager.spawn_agent(&first, "lonely").await.unwrap();
        assert!(manager.agent_task(&second, "lonely").await.is_err());
        assert!(manager
            .agent_task("no-such-session", "worker")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn per_agent_concurrency_is_bounded() {
        let manager = test_manager(16, 1);
        let session = manager.create_session().await;
        manager.spawn_agent(&session, "solo").await.unwrap();

        let _held = manager.acquire(&session, "solo").await.unwrap();
        let err = manager.acquire(&session, "solo").await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)));
        drop(_held);
        assert!(manager.acquire(&session, "solo").await.is_ok());
    }

    #[tokio::test]
    async fn global_concurrency_spans_sessions() {
        let manager = test_manager(1, 4);
        let first = manager.create_session().await;
        let second = manager.create_session().await;
        manager.spawn_agent(&first, "a").await.unwrap();
        manager.spawn_agent(&second, "b").await.unwrap();

        let _held = manager.acquire(&first, "a").await.unwrap();
        // A different session/agent still blocks on the global bound.
        let err = manager.acquire(&second, "b").await.unwrap_err();
        assert!(matches!(err, Error::ResourceUnavailable(_)));
    }

    #[tokio::test]
    async fn shared_tool_policy_is_deny_by_default() {
        let manager = test_manager(16, 4);
        let session = manager
            .create_session_with_policy(vec!["calc".to_string()])
            .await;
        assert!(manager.is_tool_allowed(&session, "calc").await.unwrap());
        assert!(!manager.is_tool_allowed(&session, "echo").await.unwrap());

        let denied = manager.create_session().await;
        assert!(!manager.is_tool_allowed(&denied, "calc").await.unwrap());

        let effective = manager
            .effective_tools(&session, &["calc".to_string(), "echo".to_string()])
            .await
            .unwrap();
        assert_eq!(effective, vec!["calc".to_string()]);
        assert!(manager.is_tool_allowed("missing", "calc").await.is_err());
    }

    #[tokio::test]
    async fn removed_sessions_release_their_agents() {
        let manager = test_manager(16, 4);
        let session = manager.create_session().await;
        manager.spawn_agent(&session, "a").await.unwrap();
        assert_eq!(manager.session_count().await, 1);
        assert!(manager.remove_session(&session).await);
        assert!(!manager.remove_session(&session).await);
        assert!(manager.agent_task(&session, "a").await.is_err());
        assert!(manager.acquire(&session, "a").await.is_err());
    }

    #[tokio::test]
    async fn idle_sessions_expire_but_active_ones_survive() {
        let manager = test_manager(16, 4);
        let idle = manager.create_session().await;
        let active = manager.create_session().await;

        // Age the idle session past the threshold with a short sleep, then
        // touch the active one so only the idle session expires.
        tokio::time::sleep(Duration::from_millis(15)).await;
        manager.spawn_agent(&active, "worker").await.unwrap();
        let removed = manager.remove_idle_sessions(Duration::from_millis(5)).await;
        assert_eq!(removed, 1);
        assert!(manager.agent_task(&idle, "worker").await.is_err());
        assert!(manager.agent_task(&active, "worker").await.is_ok());
        assert_eq!(manager.session_count().await, 1);
    }
}
