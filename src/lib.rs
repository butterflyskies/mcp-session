//! Bounded session management for MCP servers.
//!
//! This crate provides [`BoundedSessionManager`], a wrapper around rmcp's
//! [`LocalSessionManager`] that enforces:
//!
//! - **Maximum concurrent sessions** with FIFO eviction of the oldest session
//!   when the limit is reached.
//! - **Optional rate limiting** on session creation via a sliding-window
//!   counter.
//! - **Idle timeout** via rmcp's `keep_alive` configuration (passed through).
//! - **Optional max-lifetime enforcement** that closes sessions after a fixed
//!   duration regardless of activity.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use mcp_session::BoundedSessionManager;
//! use mcp_session::SessionConfig;
//!
//! let mut session_config = SessionConfig::default();
//! session_config.keep_alive = Some(std::time::Duration::from_secs(4 * 60 * 60));
//!
//! let manager = Arc::new(
//!     BoundedSessionManager::new(session_config, 100)
//!         .with_rate_limit(10, std::time::Duration::from_secs(60)),
//! );
//!
//! // Pass `manager` to `StreamableHttpService::new(factory, manager, config)`
//! ```
//!
//! # Builder usage
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use mcp_session::BoundedSessionManagerBuilder;
//!
//! let manager = BoundedSessionManagerBuilder::new(100)
//!     .idle_timeout(Duration::from_secs(4 * 60 * 60))
//!     .max_lifetime(Duration::from_secs(24 * 60 * 60))
//!     .rate_limit(10, Duration::from_secs(60))
//!     .build();
//!
//! // Pass `manager` to `StreamableHttpService::new(factory, manager, config)`
//! ```

#![warn(missing_docs)]

use std::collections::VecDeque;
use std::sync::Weak;
use std::time::{Duration, Instant};

use futures_core::Stream;
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::{
    streamable_http_server::session::{
        local::{LocalSessionManager, LocalSessionManagerError, LocalSessionWorker},
        ServerSseMessage, SessionManager,
    },
    WorkerTransport,
};

// Re-export types that consumers need so they don't have to depend on rmcp
// directly for basic session configuration.
pub use rmcp::transport::streamable_http_server::session::local::SessionConfig;
pub use rmcp::transport::streamable_http_server::session::SessionId;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`BoundedSessionManager`].
#[derive(Debug, thiserror::Error)]
pub enum BoundedSessionError {
    /// Propagated from the inner [`LocalSessionManager`].
    #[error(transparent)]
    Inner(#[from] LocalSessionManagerError),
    /// Session creation was rejected because the rate limit was exceeded.
    #[error("session creation rate limit exceeded")]
    RateLimited,
}

// ---------------------------------------------------------------------------
// CloseReason
// ---------------------------------------------------------------------------

/// The reason a session was closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    /// The session was evicted to make room for a new one (FIFO).
    Evicted,
    /// The session exceeded its configured maximum lifetime.
    MaxLifetime,
    /// The session was closed explicitly.
    Closed,
}

impl std::fmt::Display for CloseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Evicted => f.write_str("Evicted"),
            Self::MaxLifetime => f.write_str("MaxLifetime"),
            Self::Closed => f.write_str("Closed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal lifecycle tracking
// ---------------------------------------------------------------------------

struct SessionLifecycleEntry {
    created_at: Instant,
    abort_handle: Option<tokio::task::AbortHandle>,
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Sliding-window rate limiter for session creation.
struct RateLimiter {
    max_creates: usize,
    window: Duration,
    tracker: tokio::sync::Mutex<VecDeque<Instant>>,
}

impl RateLimiter {
    fn new(max_creates: usize, window: Duration) -> Self {
        Self {
            max_creates,
            window,
            tracker: tokio::sync::Mutex::new(VecDeque::new()),
        }
    }

    /// Reserve a slot. Returns `Err(BoundedSessionError::RateLimited)` if the
    /// window is full. On success, the caller **must** eventually call
    /// [`rollback`](Self::rollback) if session creation subsequently fails, to
    /// return the slot.
    async fn reserve(&self) -> Result<Instant, BoundedSessionError> {
        let mut tracker = self.tracker.lock().await;
        let now = Instant::now();
        // Prune entries that have fallen outside the window.
        while tracker
            .front()
            .is_some_and(|t| now.duration_since(*t) > self.window)
        {
            tracker.pop_front();
        }
        if tracker.len() >= self.max_creates {
            return Err(BoundedSessionError::RateLimited);
        }
        tracker.push_back(now);
        Ok(now)
    }

    /// Roll back a previously reserved slot (identified by its timestamp) when
    /// session creation fails after [`reserve`](Self::reserve) succeeds.
    async fn rollback(&self, reserved_at: Instant) {
        let mut tracker = self.tracker.lock().await;
        // Find and remove exactly one entry matching the reserved timestamp.
        // Under concurrent interleaving, a later reservation may have been
        // pushed after ours, so the entry is not necessarily at the back.
        if let Some(pos) = tracker.iter().rposition(|t| *t == reserved_at) {
            tracker.remove(pos);
        }
    }
}

// ---------------------------------------------------------------------------
// BoundedSessionManager
// ---------------------------------------------------------------------------

/// Wraps [`LocalSessionManager`] and limits the number of concurrent sessions.
///
/// When the limit is reached, the oldest session (by creation order) is closed
/// before the new one is created. This prevents unbounded memory growth when
/// many clients connect without explicitly closing their sessions.
///
/// Optionally, a rate limit can be applied to session creation via
/// [`BoundedSessionManager::with_rate_limit`].
///
/// For max-lifetime enforcement and structured lifecycle logging, prefer
/// constructing via [`BoundedSessionManagerBuilder`].
///
/// # Concurrency note
///
/// Under concurrent session creation, the live count may transiently exceed
/// `max_sessions` by at most the number of concurrent callers. The limit is
/// best-effort under contention; use a semaphore if exact enforcement is
/// required.
pub struct BoundedSessionManager {
    inner: LocalSessionManager,
    max_sessions: usize,
    /// Tracks session IDs in creation order for FIFO eviction.
    creation_order: tokio::sync::Mutex<VecDeque<SessionId>>,
    /// Optional sliding-window rate limiter for session creation.
    rate_limiter: Option<RateLimiter>,
    /// Optional maximum session lifetime. Sessions are closed after this
    /// duration regardless of activity.
    max_lifetime: Option<Duration>,
    /// Per-session lifecycle entries (created_at + abort handle for the
    /// max-lifetime timer task).
    lifecycle: tokio::sync::RwLock<std::collections::HashMap<SessionId, SessionLifecycleEntry>>,
    /// Weak self-reference, populated only when max_lifetime is Some so that
    /// the timer task can call back into the manager.
    self_ref: Option<Weak<Self>>,
}

impl BoundedSessionManager {
    /// Create a new `BoundedSessionManager`.
    ///
    /// * `session_config` — passed through to the inner [`LocalSessionManager`].
    /// * `max_sessions`   — maximum number of concurrent sessions. When this
    ///   limit is reached, the oldest session is evicted before creating a new
    ///   one. Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if `max_sessions` is 0.
    pub fn new(session_config: SessionConfig, max_sessions: usize) -> Self {
        assert!(max_sessions >= 1, "max_sessions must be at least 1, got 0");
        let mut inner = LocalSessionManager::default();
        inner.session_config = session_config;
        Self {
            inner,
            max_sessions,
            creation_order: tokio::sync::Mutex::new(VecDeque::new()),
            rate_limiter: None,
            max_lifetime: None,
            lifecycle: tokio::sync::RwLock::new(std::collections::HashMap::new()),
            self_ref: None,
        }
    }

    /// Configure a rate limit on session creation.
    ///
    /// At most `max_creates` sessions may be created within any rolling
    /// `window` duration. If exceeded, [`BoundedSessionError::RateLimited`] is
    /// returned and no eviction is performed.
    ///
    /// # Panics
    ///
    /// Panics if `max_creates` is 0. Pass no rate limit instead of 0 — a limit
    /// of zero would silently block all session creation.
    #[must_use]
    pub fn with_rate_limit(mut self, max_creates: usize, window: Duration) -> Self {
        assert!(
            max_creates >= 1,
            "max_creates must be at least 1; pass no rate limit instead of 0"
        );
        self.rate_limiter = Some(RateLimiter::new(max_creates, window));
        self
    }

    /// Returns the number of currently tracked live sessions.
    pub async fn active_session_count(&self) -> usize {
        self.lifecycle.read().await.len()
    }

    /// Unified close path: remove lifecycle entry, cancel timer, close inner
    /// session, update creation order, and log with the given reason.
    async fn close_session_impl(
        &self,
        id: &SessionId,
        reason: CloseReason,
    ) -> Result<(), BoundedSessionError> {
        let entry = self.lifecycle.write().await.remove(id);
        if let Some(ref e) = entry {
            if let Some(ref handle) = e.abort_handle {
                handle.abort();
            }
        }
        let elapsed = entry.as_ref().map(|e| e.created_at.elapsed());

        let close_result = self.inner.close_session(id).await;

        let mut order = self.creation_order.lock().await;
        order.retain(|s| s != id);
        drop(order);

        if let Some(elapsed) = elapsed {
            tracing::info!(
                session_id = %id,
                duration_secs = elapsed.as_secs_f64(),
                reason = %reason,
                "session closed"
            );
        }

        close_result.map_err(Into::into)
    }
}

impl SessionManager for BoundedSessionManager {
    type Error = BoundedSessionError;
    type Transport = WorkerTransport<LocalSessionWorker>;

    async fn create_session(&self) -> Result<(SessionId, Self::Transport), Self::Error> {
        // ----------------------------------------------------------------
        // Critical section 1: rate-limit check.
        // ----------------------------------------------------------------
        let rate_reserved_at = if let Some(ref limiter) = self.rate_limiter {
            Some(limiter.reserve().await?)
        } else {
            None
        };

        // ----------------------------------------------------------------
        // Determine eviction candidate (short critical section).
        // ----------------------------------------------------------------
        let evict_candidate = {
            let order = self.creation_order.lock().await;
            // Use the inner sessions map for the authoritative live count so
            // that expired sessions (which are removed from inner but remain
            // in the deque) do not consume a capacity slot.
            let live_count = self.inner.sessions.read().await.len();
            if live_count >= self.max_sessions {
                order.front().cloned()
            } else {
                None
            }
        };

        // ----------------------------------------------------------------
        // Evict oldest (no lock held across this await).
        // ----------------------------------------------------------------
        if let Some(ref oldest) = evict_candidate {
            let _ = self.close_session_impl(oldest, CloseReason::Evicted).await;
        }

        // ----------------------------------------------------------------
        // Create new session (no lock held across this await).
        // ----------------------------------------------------------------
        let result = self.inner.create_session().await;

        // Roll back the rate-limit slot if creation failed.
        if result.is_err() {
            if let (Some(ref limiter), Some(reserved_at)) = (&self.rate_limiter, rate_reserved_at) {
                limiter.rollback(reserved_at).await;
            }
        }

        let (id, transport) = result?;

        // ----------------------------------------------------------------
        // Critical section 2: update the creation-order deque.
        // ----------------------------------------------------------------
        {
            let mut order = self.creation_order.lock().await;
            // Remove the evicted entry if it's still present.
            if let Some(ref oldest) = evict_candidate {
                order.retain(|s| s != oldest);
            }
            // Prune any deque entries for sessions that are no longer live
            // (handles the drift caused by keep_alive expiry: finding #4).
            let live_ids: std::collections::HashSet<_> = {
                // Snapshot the live session IDs without holding two locks
                // simultaneously (creation_order lock is already held here;
                // sessions is a RwLock so a read lock is fine).
                self.inner.sessions.read().await.keys().cloned().collect()
            };
            order.retain(|s| live_ids.contains(s));
            order.push_back(id.clone());
        }

        // ----------------------------------------------------------------
        // Register lifecycle entry (and spawn max-lifetime timer if needed).
        // ----------------------------------------------------------------
        let abort_handle =
            if let (Some(max_lifetime), Some(ref weak)) = (self.max_lifetime, &self.self_ref) {
                let weak = weak.clone();
                let timer_id = id.clone();
                let handle = tokio::spawn(async move {
                    tokio::time::sleep(max_lifetime).await;
                    if let Some(arc) = weak.upgrade() {
                        let _ = arc
                            .close_session_impl(&timer_id, CloseReason::MaxLifetime)
                            .await;
                    }
                });
                Some(handle.abort_handle())
            } else {
                None
            };

        self.lifecycle.write().await.insert(
            id.clone(),
            SessionLifecycleEntry {
                created_at: Instant::now(),
                abort_handle,
            },
        );

        tracing::info!(session_id = %id, "session created");

        Ok((id, transport))
    }

    async fn close_session(&self, id: &SessionId) -> Result<(), Self::Error> {
        self.close_session_impl(id, CloseReason::Closed).await
    }

    async fn initialize_session(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<ServerJsonRpcMessage, Self::Error> {
        self.inner
            .initialize_session(id, message)
            .await
            .map_err(Into::into)
    }

    async fn has_session(&self, id: &SessionId) -> Result<bool, Self::Error> {
        self.inner.has_session(id).await.map_err(Into::into)
    }

    async fn create_stream(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        self.inner
            .create_stream(id, message)
            .await
            .map_err(Into::into)
    }

    async fn accept_message(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<(), Self::Error> {
        self.inner
            .accept_message(id, message)
            .await
            .map_err(Into::into)
    }

    async fn create_standalone_stream(
        &self,
        id: &SessionId,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        self.inner
            .create_standalone_stream(id)
            .await
            .map_err(Into::into)
    }

    async fn resume(
        &self,
        id: &SessionId,
        last_event_id: String,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        self.inner
            .resume(id, last_event_id)
            .await
            .map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// BoundedSessionManagerBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`BoundedSessionManager`].
///
/// Prefer this over [`BoundedSessionManager::new`] when you need max-lifetime
/// enforcement or want to configure idle timeouts without manually constructing
/// a [`SessionConfig`].
///
/// # Example
///
/// ```rust,no_run
/// use std::time::Duration;
/// use mcp_session::BoundedSessionManagerBuilder;
///
/// let manager = BoundedSessionManagerBuilder::new(100)
///     .idle_timeout(Duration::from_secs(4 * 60 * 60))
///     .max_lifetime(Duration::from_secs(24 * 60 * 60))
///     .rate_limit(10, Duration::from_secs(60))
///     .build();
/// ```
pub struct BoundedSessionManagerBuilder {
    max_sessions: usize,
    idle_timeout: Option<Duration>,
    max_lifetime: Option<Duration>,
    session_config: Option<SessionConfig>,
    rate_limit: Option<(usize, Duration)>,
}

impl BoundedSessionManagerBuilder {
    /// Create a new builder.
    ///
    /// # Panics
    ///
    /// Panics if `max_sessions` is 0.
    pub fn new(max_sessions: usize) -> Self {
        assert!(max_sessions >= 1, "max_sessions must be at least 1, got 0");
        Self {
            max_sessions,
            idle_timeout: None,
            max_lifetime: None,
            session_config: None,
            rate_limit: None,
        }
    }

    /// Set the idle timeout (maps to `SessionConfig::keep_alive`).
    ///
    /// If a `session_config` with `keep_alive` already set is also provided,
    /// this value takes precedence and a warning is logged.
    #[must_use]
    pub fn idle_timeout(mut self, duration: Duration) -> Self {
        self.idle_timeout = Some(duration);
        self
    }

    /// Set a hard maximum lifetime for sessions.
    ///
    /// Sessions are closed after this duration regardless of activity. Requires
    /// a Tokio runtime to be active when [`build`](Self::build) is called.
    #[must_use]
    pub fn max_lifetime(mut self, duration: Duration) -> Self {
        self.max_lifetime = Some(duration);
        self
    }

    /// Supply a fully constructed [`SessionConfig`].
    ///
    /// If [`idle_timeout`](Self::idle_timeout) is also set, it overrides
    /// `keep_alive` in this config.
    #[must_use]
    pub fn session_config(mut self, config: SessionConfig) -> Self {
        self.session_config = Some(config);
        self
    }

    /// Configure a sliding-window rate limit on session creation.
    ///
    /// At most `max_creates` sessions may be created in any rolling `window`.
    ///
    /// # Panics
    ///
    /// Panics if `max_creates` is 0.
    #[must_use]
    pub fn rate_limit(mut self, max_creates: usize, window: Duration) -> Self {
        assert!(
            max_creates >= 1,
            "max_creates must be at least 1; pass no rate limit instead of 0"
        );
        self.rate_limit = Some((max_creates, window));
        self
    }

    /// Build the [`BoundedSessionManager`], returning an [`Arc`](std::sync::Arc).
    pub fn build(self) -> std::sync::Arc<BoundedSessionManager> {
        let explicit_config = self.session_config.is_some();
        let mut session_config = self.session_config.unwrap_or_default();

        if let Some(idle) = self.idle_timeout {
            if explicit_config && session_config.keep_alive.is_some() {
                tracing::warn!("idle_timeout overrides SessionConfig::keep_alive");
            }
            session_config.keep_alive = Some(idle);
        }

        let max_sessions = self.max_sessions;
        let max_lifetime = self.max_lifetime;
        let rate_limiter = self
            .rate_limit
            .map(|(max_creates, window)| RateLimiter::new(max_creates, window));

        std::sync::Arc::new_cyclic(|weak| {
            let mut inner = LocalSessionManager::default();
            inner.session_config = session_config;
            BoundedSessionManager {
                inner,
                max_sessions,
                creation_order: tokio::sync::Mutex::new(VecDeque::new()),
                rate_limiter,
                max_lifetime,
                lifecycle: tokio::sync::RwLock::new(std::collections::HashMap::new()),
                self_ref: Some(weak.clone()),
            }
        })
    }
}
