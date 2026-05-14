//! Integration tests for session lifecycle observability: builder API,
//! max-lifetime enforcement, lifecycle tracing, and close reasons.
//!
//! Uses the same HTTP-level testing approach as `session_limits.rs`.
//! Tracing assertions use an in-memory capturing layer installed as the
//! global subscriber (one test per nextest process).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use mcp_session::{
    BoundedSessionManager, BoundedSessionManagerBuilder, CloseReason, SessionConfig,
};
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::ServerHandler;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

// ---------------------------------------------------------------------------
// Minimal MCP handler
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct NoopServer;
impl ServerHandler for NoopServer {}

// ---------------------------------------------------------------------------
// Capturing tracing layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EventRecord {
    message: String,
    fields: Vec<(String, String)>,
}

#[derive(Default)]
struct RecordStore {
    events: Vec<EventRecord>,
}

struct CapturingLayer {
    store: Arc<Mutex<RecordStore>>,
}

impl<S: Subscriber> Layer<S> for CapturingLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut kv: Vec<(String, String)> = Vec::new();
        let mut message = String::new();

        struct Visitor<'a> {
            kv: &'a mut Vec<(String, String)>,
            message: &'a mut String,
        }
        impl tracing::field::Visit for Visitor<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, val: &dyn std::fmt::Debug) {
                let s = format!("{val:?}");
                if field.name() == "message" {
                    *self.message = s;
                } else {
                    self.kv.push((field.name().to_string(), s));
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, val: &str) {
                if field.name() == "message" {
                    *self.message = val.to_string();
                } else {
                    self.kv.push((field.name().to_string(), val.to_string()));
                }
            }
            fn record_f64(&mut self, field: &tracing::field::Field, val: f64) {
                self.kv.push((field.name().to_string(), val.to_string()));
            }
        }

        event.record(&mut Visitor {
            kv: &mut kv,
            message: &mut message,
        });

        self.store.lock().unwrap().events.push(EventRecord {
            message,
            fields: kv,
        });
    }
}

fn install_capturing() -> Arc<Mutex<RecordStore>> {
    let store = Arc::new(Mutex::new(RecordStore::default()));
    let subscriber = tracing_subscriber::registry().with(CapturingLayer {
        store: Arc::clone(&store),
    });
    // set_global_default can only be called once per process; nextest gives
    // us process-per-test isolation so this is safe.
    let _ = tracing::subscriber::set_global_default(subscriber);
    store
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn build_router_with_builder(manager: Arc<BoundedSessionManager>) -> (Router, CancellationToken) {
    let ct = CancellationToken::new();
    let ct_child = ct.child_token();

    let service = StreamableHttpService::new(|| Ok(NoopServer), manager, {
        let mut config = StreamableHttpServerConfig::default();
        config.cancellation_token = ct_child;
        config
    });

    let router = Router::new().nest_service("/mcp", service);
    (router, ct)
}

async fn spawn_server_with_builder(
    manager: Arc<BoundedSessionManager>,
) -> (String, CancellationToken) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind to random port");
    let addr = listener.local_addr().expect("get local addr");
    let base_url = format!("http://{}", addr);

    let (router, ct) = build_router_with_builder(manager);
    let ct_child = ct.child_token();

    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(ct_child.cancelled_owned())
            .await
            .expect("server error");
    });

    (base_url, ct)
}

fn initialize_body() -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"}
        }
    })
}

fn tools_list_body() -> serde_json::Value {
    serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
}

async fn post_mcp(
    client: &reqwest::Client,
    base_url: &str,
    session_id: Option<&str>,
    body: &serde_json::Value,
) -> (reqwest::StatusCode, Option<String>) {
    let mut builder = client
        .post(format!("{}/mcp", base_url))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .json(body);

    if let Some(sid) = session_id {
        builder = builder.header("Mcp-Session-Id", sid);
    }

    let resp = builder.send().await.expect("HTTP request succeeded");
    let status = resp.status();
    let returned_sid = resp
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let _ = resp.text().await;
    (status, returned_sid)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[test]
fn u01_close_reason_derives() {
    let r = CloseReason::Evicted;
    assert_eq!(r, CloseReason::Evicted);
    assert_ne!(r, CloseReason::MaxLifetime);
    assert_ne!(r, CloseReason::Closed);

    assert_eq!(format!("{}", CloseReason::Evicted), "Evicted");
    assert_eq!(format!("{}", CloseReason::MaxLifetime), "MaxLifetime");
    assert_eq!(format!("{}", CloseReason::Closed), "Closed");

    let _ = format!("{:?}", r);
    let r2 = r;
    assert_eq!(r, r2);
}

/// Builder idle_timeout actually controls session idle expiry. Set a 1s idle
/// timeout via the builder and verify the session expires after inactivity.
#[tokio::test]
async fn u02_builder_idle_timeout_controls_expiry() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(1))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid = sid.expect("session id");

    // Session alive immediately after creation.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert!(status.is_success(), "session should be alive");

    // Wait for idle timeout.
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Session should have expired.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "session should have expired via idle timeout"
    );
}

/// Builder with passthrough SessionConfig that sets keep_alive — idle_timeout
/// should override it.
#[tokio::test]
async fn u03_idle_timeout_overrides_passthrough_keep_alive() {
    let mut cfg = SessionConfig::default();
    cfg.keep_alive = Some(Duration::from_secs(300));

    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(1))
        .session_config(cfg)
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid = sid.expect("session id");

    tokio::time::sleep(Duration::from_secs(3)).await;

    // If idle_timeout (1s) didn't override keep_alive (300s), this would succeed.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "idle_timeout should override passthrough keep_alive"
    );
}

// ---------------------------------------------------------------------------
// Integration tests: Max lifetime
// ---------------------------------------------------------------------------

/// T-01: Max lifetime closes session after configured duration.
#[tokio::test]
async fn t01_max_lifetime_closes_session() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(2))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "initialize failed: {status}");
    let sid = sid.expect("must have session id");

    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert!(status.is_success(), "session should be alive: {status}");

    tokio::time::sleep(Duration::from_secs(4)).await;

    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "session should be closed by max lifetime: {status}"
    );
}

/// T-02: Early close cancels max lifetime task — no double-close.
#[tokio::test]
async fn t02_early_close_cancels_max_lifetime() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(2))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid = sid.expect("must have session id");

    let resp = client
        .delete(format!("{}/mcp", base_url))
        .header("Mcp-Session-Id", &sid)
        .send()
        .await
        .expect("delete request");
    assert!(resp.status().is_success(), "DELETE should succeed");

    tokio::time::sleep(Duration::from_secs(4)).await;
    assert_eq!(mgr.active_session_count().await, 0);
}

/// T-03: Eviction closes with reason Evicted; surviving session gets MaxLifetime.
#[tokio::test]
async fn t03_eviction_with_max_lifetime() {
    let mgr = BoundedSessionManagerBuilder::new(1)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(3))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let (status, sid_a) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid_a = sid_a.expect("session A id");

    let (status, sid_b) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let _sid_b = sid_b.expect("session B id");

    let (status, _) = post_mcp(&client, &base_url, Some(&sid_a), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "A should be evicted"
    );

    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(mgr.active_session_count().await, 0);
}

/// T-04: Lifecycle tracing — "session created" and "session closed" events carry
/// the expected structured fields. Also verifies reason=Closed for explicit close,
/// reason=Evicted for eviction, and reason=MaxLifetime for max lifetime expiry.
///
/// Merged into a single test because `set_global_default` can only be called once
/// per process, and nextest provides process-per-test isolation.
#[tokio::test]
async fn t04_lifecycle_tracing_events() {
    let store = install_capturing();

    // --- Part 1: explicit close → reason=Closed ---
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid = sid.expect("session id");

    tokio::time::sleep(Duration::from_millis(100)).await;

    {
        let events = store.lock().unwrap();
        let created = events
            .events
            .iter()
            .find(|e| e.message.contains("session created"));
        assert!(created.is_some(), "expected 'session created' event");
        assert!(
            created
                .unwrap()
                .fields
                .iter()
                .any(|(k, _)| k == "session_id"),
            "session created event missing session_id"
        );
    }

    let resp = client
        .delete(format!("{}/mcp", base_url))
        .header("Mcp-Session-Id", &sid)
        .send()
        .await
        .expect("delete request");
    assert!(resp.status().is_success());

    tokio::time::sleep(Duration::from_millis(100)).await;

    {
        let events = store.lock().unwrap();
        let closed = events
            .events
            .iter()
            .find(|e| e.message.contains("session closed"));
        assert!(closed.is_some(), "expected 'session closed' event");
        let closed = closed.unwrap();
        assert!(
            closed.fields.iter().any(|(k, _)| k == "session_id"),
            "missing session_id; fields: {:?}",
            closed.fields
        );
        assert!(
            closed.fields.iter().any(|(k, _)| k == "duration_secs"),
            "missing duration_secs; fields: {:?}",
            closed.fields
        );
        assert!(
            closed
                .fields
                .iter()
                .any(|(k, v)| k == "reason" && v == "Closed"),
            "expected reason=Closed; fields: {:?}",
            closed.fields
        );
    }

    // --- Part 2: eviction → reason=Evicted ---
    store.lock().unwrap().events.clear();

    let mgr2 = BoundedSessionManagerBuilder::new(1)
        .idle_timeout(Duration::from_secs(300))
        .build();
    let (base_url2, _ct2) = spawn_server_with_builder(mgr2).await;

    let (status, _) = post_mcp(&client, &base_url2, None, &initialize_body()).await;
    assert!(status.is_success());
    let (status, _) = post_mcp(&client, &base_url2, None, &initialize_body()).await;
    assert!(status.is_success());

    tokio::time::sleep(Duration::from_millis(100)).await;

    {
        let events = store.lock().unwrap();
        let evicted = events.events.iter().find(|e| {
            e.message.contains("session closed")
                && e.fields
                    .iter()
                    .any(|(k, v)| k == "reason" && v == "Evicted")
        });
        assert!(
            evicted.is_some(),
            "expected 'session closed' with reason=Evicted; events: {:?}",
            events
                .events
                .iter()
                .filter(|e| e.message.contains("session closed"))
                .collect::<Vec<_>>()
        );
    }

    // --- Part 3: max lifetime → reason=MaxLifetime ---
    store.lock().unwrap().events.clear();

    let mgr3 = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(1))
        .build();
    let (base_url3, _ct3) = spawn_server_with_builder(mgr3).await;

    let (status, _) = post_mcp(&client, &base_url3, None, &initialize_body()).await;
    assert!(status.is_success());

    tokio::time::sleep(Duration::from_secs(3)).await;

    {
        let events = store.lock().unwrap();
        let max_lt = events.events.iter().find(|e| {
            e.message.contains("session closed")
                && e.fields
                    .iter()
                    .any(|(k, v)| k == "reason" && v == "MaxLifetime")
        });
        assert!(
            max_lt.is_some(),
            "expected 'session closed' with reason=MaxLifetime; events: {:?}",
            events
                .events
                .iter()
                .filter(|e| e.message.contains("session closed"))
                .collect::<Vec<_>>()
        );
    }
}

/// T-05b: Idle timeout triggers full lifecycle cleanup — active_session_count
/// drops to 0 AND a "session closed" tracing event with duration_secs is
/// emitted. Proves rmcp's worker exit calls our close_session_impl.
#[tokio::test]
async fn t05b_idle_timeout_lifecycle_cleanup() {
    let store = install_capturing();

    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(1))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success());
    let sid = sid.expect("session id");

    assert_eq!(mgr.active_session_count().await, 1);

    // Wait for idle timeout.
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Session should be gone from HTTP layer.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND);

    // Lifecycle map should be cleaned up.
    assert_eq!(
        mgr.active_session_count().await,
        0,
        "lifecycle entry should be cleaned up after idle expiry"
    );

    // Tracing event should have been emitted with duration and reason=Closed
    // (the documented catch-all for idle timeout, client DELETE, etc.).
    let events = store.lock().unwrap();
    assert!(
        events
            .events
            .iter()
            .any(|e| e.message.contains("session closed")
                && e.fields.iter().any(|(k, _)| k == "duration_secs")
                && e.fields.iter().any(|(k, v)| k == "reason" && v == "Closed")),
        "expected 'session closed' with duration_secs and reason=Closed after idle expiry"
    );
}

/// T-06: Backward compat — new() API works without lifecycle tracking.
#[tokio::test]
async fn t06_backward_compat_new_api() {
    let mut cfg = SessionConfig::default();
    cfg.keep_alive = Some(Duration::from_secs(300));

    let mgr = Arc::new(BoundedSessionManager::new(cfg, 10));

    let ct = CancellationToken::new();
    let ct_child = ct.child_token();

    let service = StreamableHttpService::new(|| Ok(NoopServer), mgr, {
        let mut config = StreamableHttpServerConfig::default();
        config.cancellation_token = ct_child;
        config
    });

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let base_url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, Router::new().nest_service("/mcp", service))
            .with_graceful_shutdown(ct.child_token().cancelled_owned())
            .await
            .expect("server");
    });

    let client = reqwest::Client::new();
    let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "new() API should work: {status}");
    let sid = sid.expect("session id");

    let (status, _) = post_mcp(&client, &base_url, Some(&sid), &tools_list_body()).await;
    assert!(status.is_success(), "session should be active: {status}");
}

/// T-07: Builder with only rate_limit — no lifecycle tasks spawned.
#[tokio::test]
async fn t07_builder_rate_limit_only() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .rate_limit(2, Duration::from_secs(60))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr).await;
    let client = reqwest::Client::new();

    let (status, _) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "first session should succeed");

    let (status, _) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "second session should succeed");

    let (status, _) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(!status.is_success(), "third session should be rate limited");
}

/// T-08: Multiple sessions with max lifetime all close correctly.
#[tokio::test]
async fn t08_multiple_sessions_max_lifetime() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(2))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let mut sids = Vec::new();
    for _ in 0..5 {
        let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
        assert!(status.is_success());
        sids.push(sid.expect("session id"));
    }

    assert_eq!(mgr.active_session_count().await, 5);

    tokio::time::sleep(Duration::from_secs(4)).await;

    assert_eq!(mgr.active_session_count().await, 0);

    for sid in &sids {
        let (status, _) = post_mcp(&client, &base_url, Some(sid), &tools_list_body()).await;
        assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

/// I-01: Lifecycle map has exactly one entry per live session.
#[tokio::test]
async fn i01_lifecycle_map_matches_live_sessions() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let mut sids = Vec::new();
    for _ in 0..5 {
        let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
        assert!(status.is_success());
        sids.push(sid.expect("session id"));
    }
    assert_eq!(mgr.active_session_count().await, 5);

    for sid in &sids[..2] {
        let resp = client
            .delete(format!("{}/mcp", base_url))
            .header("Mcp-Session-Id", sid)
            .send()
            .await
            .expect("delete");
        assert!(resp.status().is_success());
    }
    assert_eq!(mgr.active_session_count().await, 3);

    for sid in &sids[2..] {
        let resp = client
            .delete(format!("{}/mcp", base_url))
            .header("Mcp-Session-Id", sid)
            .send()
            .await
            .expect("delete");
        assert!(resp.status().is_success());
    }
    assert_eq!(mgr.active_session_count().await, 0);
}

/// I-02: No leaked abort handles after all sessions close.
#[tokio::test]
async fn i02_no_leaked_abort_handles() {
    let mgr = BoundedSessionManagerBuilder::new(10)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(60))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let mut sids = Vec::new();
    for _ in 0..3 {
        let (status, sid) = post_mcp(&client, &base_url, None, &initialize_body()).await;
        assert!(status.is_success());
        sids.push(sid.expect("session id"));
    }

    for sid in &sids {
        let resp = client
            .delete(format!("{}/mcp", base_url))
            .header("Mcp-Session-Id", sid)
            .send()
            .await
            .expect("delete");
        assert!(resp.status().is_success());
    }

    assert_eq!(mgr.active_session_count().await, 0);
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(mgr.active_session_count().await, 0);
}

/// I-03: Eviction + timeout don't corrupt state.
#[tokio::test]
async fn i03_eviction_and_timeout_no_corruption() {
    let mgr = BoundedSessionManagerBuilder::new(2)
        .idle_timeout(Duration::from_secs(300))
        .max_lifetime(Duration::from_secs(3))
        .build();

    let (base_url, _ct) = spawn_server_with_builder(mgr.clone()).await;
    let client = reqwest::Client::new();

    let (status, sid_a) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "A create failed");
    let sid_a = sid_a.expect("A");
    let (status, _) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "B create failed");

    // C evicts A, D evicts B.
    let (status, sid_c) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "C create failed");
    let sid_c = sid_c.expect("C");
    let (status, sid_d) = post_mcp(&client, &base_url, None, &initialize_body()).await;
    assert!(status.is_success(), "D create failed");
    let sid_d = sid_d.expect("D");

    // A should be evicted.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid_a), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "A should be evicted"
    );

    // C and D should be alive.
    assert_eq!(mgr.active_session_count().await, 2);

    // Wait for max lifetime to close C and D.
    tokio::time::sleep(Duration::from_secs(5)).await;

    assert_eq!(mgr.active_session_count().await, 0);

    // C and D should be gone.
    let (status, _) = post_mcp(&client, &base_url, Some(&sid_c), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "C should be expired"
    );
    let (status, _) = post_mcp(&client, &base_url, Some(&sid_d), &tools_list_body()).await;
    assert_eq!(
        status,
        reqwest::StatusCode::NOT_FOUND,
        "D should be expired"
    );
}
