use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::sync::watch;

use crate::compile_cache::CompileCacheStore;

#[derive(Clone)]
struct HealthState {
    leader_rx: watch::Receiver<bool>,
    compile_cache: Option<CompileCacheStore>,
}

pub fn build_health_router(
    leader_rx: watch::Receiver<bool>,
    compile_cache: Option<CompileCacheStore>,
) -> Router {
    let state = HealthState {
        leader_rx,
        compile_cache,
    };

    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/compile-cache-stats", get(compile_cache_stats))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz(State(state): State<HealthState>) -> Response {
    if *state.leader_rx.borrow() {
        (
            StatusCode::OK,
            axum::Json(serde_json::json!({"status": "leader", "ready": true})),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"status": "standby", "ready": false})),
        )
            .into_response()
    }
}

async fn compile_cache_stats(State(state): State<HealthState>) -> Response {
    let Some(ref store) = state.compile_cache else {
        return (StatusCode::NOT_FOUND, "compile cache not enabled").into_response();
    };
    axum::Json(store.stats()).into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tokio::sync::watch;
    use tower::ServiceExt;

    use crate::hf_api::build_health_router;

    #[tokio::test]
    async fn healthz_always_200() {
        let (_, rx) = watch::channel(false);
        let router = build_health_router(rx, None);

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
    }

    #[tokio::test]
    async fn readyz_503_when_not_leader() {
        let (_, rx) = watch::channel(false);
        let router = build_health_router(rx, None);

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 503);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "standby");
    }

    #[tokio::test]
    async fn readyz_200_when_leader() {
        let (tx, rx) = watch::channel(false);
        let router = build_health_router(rx, None);

        tx.send(true).unwrap();

        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "leader");
    }
}
