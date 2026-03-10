use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::Router;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::debug;

use crate::compile_cache::CompileCacheStore;
use crate::nixl_vram_store::ModelPeerStore;

/// Shared state for the internal API handlers (NIXL metadata exchange + compile cache).
#[derive(Clone)]
struct InternalState {
    model_peer_store: ModelPeerStore,
    compile_cache: Option<CompileCacheStore>,
    ready: Arc<AtomicBool>,
}

/// Build the internal axum router (peer-to-peer NIXL metadata exchange + compile cache).
///
/// Endpoints:
/// - `GET /internal/nixl-vram/:agent_name` - NIXL VRAM metadata (peer fetch)
/// - `GET /internal/compile-cache/:key` - compile cache blob (peer fetch)
/// - `GET /health` - basic liveness check
/// - `GET /healthz` - readiness probe (returns 503 until daemon init is complete)
pub fn build_internal_router(
    model_peer_store: ModelPeerStore,
    compile_cache: Option<CompileCacheStore>,
    ready: Arc<AtomicBool>,
) -> Router {
    let state = InternalState {
        model_peer_store,
        compile_cache,
        ready,
    };

    Router::new()
        .route(
            "/internal/nixl-vram/{agent_name}",
            get(get_nixl_vram_metadata),
        )
        .route(
            "/internal/compile-cache/{key}",
            get(get_compile_cache_entry),
        )
        .route(
            "/internal/compile-cache-stats",
            get(get_compile_cache_stats),
        )
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// Readiness probe: returns 200 once the daemon has finished initialization,
/// 503 while it is still starting up.
async fn healthz(State(state): State<InternalState>) -> Response {
    if state.ready.load(Ordering::Relaxed) {
        (
            StatusCode::OK,
            axum::Json(serde_json::json!({"status": "ok"})),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"status": "not_ready"})),
        )
            .into_response()
    }
}

/// `GET /internal/nixl-vram/:agent_name`
///
/// Returns the raw NIXL VRAM metadata bytes for a local agent.
/// Peers call this to fetch the actual metadata after discovering
/// the lightweight pointer via the NodeCache CRD.
async fn get_nixl_vram_metadata(
    State(state): State<InternalState>,
    Path(agent_name): Path<String>,
) -> Response {
    match state.model_peer_store.get(&agent_name).await {
        Some(metadata) => {
            debug!(
                agent_name,
                len = metadata.len(),
                "serving NIXL VRAM metadata"
            );
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .body(Body::from(metadata))
                .unwrap()
        }
        None => {
            debug!(agent_name, "NIXL VRAM metadata not found");
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::empty())
                .unwrap()
        }
    }
}

/// `GET /internal/compile-cache/:key`
///
/// Returns a compile cache blob for peer-to-peer transfer.
/// Only serves from local memory/disk, never fans out to other peers
/// (that would cause infinite recursion).
async fn get_compile_cache_entry(
    State(state): State<InternalState>,
    Path(key): Path<String>,
) -> Response {
    let Some(ref store) = state.compile_cache else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap();
    };

    match store.get_local(&key).await {
        Some(data) => {
            debug!(key, len = data.len(), "serving compile cache entry");
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .body(Body::from(data))
                .unwrap()
        }
        None => {
            debug!(key, "compile cache entry not found");
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::empty())
                .unwrap()
        }
    }
}

/// `GET /internal/compile-cache-stats`
///
/// Returns JSON hit/miss counters for compile cache observability.
async fn get_compile_cache_stats(State(state): State<InternalState>) -> Response {
    let Some(ref store) = state.compile_cache else {
        return (StatusCode::NOT_FOUND, "compile cache not enabled").into_response();
    };
    axum::Json(store.stats()).into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::hf_api::build_internal_router;
    use crate::nixl_vram_store::ModelPeerStore;

    #[tokio::test]
    async fn healthz_returns_503_when_not_ready() {
        let ready = Arc::new(AtomicBool::new(false));
        let router = build_internal_router(ModelPeerStore::new(), None, Arc::clone(&ready));

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 503);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "not_ready");
    }

    #[tokio::test]
    async fn healthz_returns_200_when_ready() {
        let ready = Arc::new(AtomicBool::new(false));
        let router = build_internal_router(ModelPeerStore::new(), None, Arc::clone(&ready));

        // Flip ready to true
        ready.store(true, Ordering::Relaxed);

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }
}
