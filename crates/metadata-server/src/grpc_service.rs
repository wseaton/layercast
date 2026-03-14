use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::proto;
use crate::proto::layercast_service_server::LayercastService;
use crate::session::SessionHandler;
use crate::state::ServerState;

pub struct LayercastGrpcServer {
    state: Arc<ServerState>,
    sessions: RwLock<HashMap<String, Arc<tokio::sync::Mutex<SessionHandler>>>>,
    hf_api: hf_hub::api::tokio::Api,
    peer_discovery_timeout: Duration,
}

impl LayercastGrpcServer {
    pub fn new(
        state: Arc<ServerState>,
        hf_api: hf_hub::api::tokio::Api,
        peer_discovery_timeout: Duration,
    ) -> Self {
        Self {
            state,
            sessions: RwLock::new(HashMap::new()),
            hf_api,
            peer_discovery_timeout,
        }
    }

    async fn get_or_create_session(
        &self,
        pod_name: &str,
        pod_ip: &str,
    ) -> Arc<tokio::sync::Mutex<SessionHandler>> {
        {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(pod_name) {
                return Arc::clone(session);
            }
        }

        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get(pod_name) {
            return Arc::clone(session);
        }

        info!(pod_name, pod_ip, "creating new gRPC session");
        let handler = SessionHandler::new(
            Arc::clone(&self.state),
            pod_name.to_string(),
            pod_ip.to_string(),
            self.hf_api.clone(),
            self.peer_discovery_timeout,
        );
        let session = Arc::new(tokio::sync::Mutex::new(handler));
        sessions.insert(pod_name.to_string(), Arc::clone(&session));
        session
    }

    pub async fn remove_session(&self, pod_name: &str) {
        let removed = self.sessions.write().await.remove(pod_name);
        if let Some(session) = removed {
            let mut handler = session.lock().await;
            handler.cleanup_on_disconnect().await;
        }
    }
}

#[tonic::async_trait]
impl LayercastService for LayercastGrpcServer {
    async fn prepare_model(
        &self,
        request: Request<proto::PrepareModelRequest>,
    ) -> Result<Response<proto::PrepareModelResponse>, Status> {
        let req = request.into_inner();
        if req.pod_name.is_empty() {
            return Err(Status::invalid_argument("pod_name is required"));
        }
        let prepare = req
            .prepare
            .ok_or_else(|| Status::invalid_argument("prepare field is required"))?;

        let session = self.get_or_create_session(&req.pod_name, &req.pod_ip).await;
        let mut handler = session.lock().await;

        let timeout_override = req
            .peer_discovery_timeout_s
            .map(|s| Duration::from_secs(s as u64));
        match handler.handle_prepare(prepare, timeout_override).await {
            Ok(prepared) => Ok(Response::new(proto::PrepareModelResponse {
                prepared: Some(prepared),
            })),
            Err(msg) => Err(Status::failed_precondition(msg)),
        }
    }

    async fn model_loaded(
        &self,
        request: Request<proto::ModelLoadedRequest>,
    ) -> Result<Response<proto::ModelLoadedResponse>, Status> {
        let req = request.into_inner();
        if req.pod_name.is_empty() {
            return Err(Status::invalid_argument("pod_name is required"));
        }
        let loaded = req
            .loaded
            .ok_or_else(|| Status::invalid_argument("loaded field is required"))?;

        let session = self.get_or_create_session(&req.pod_name, "").await;
        let mut handler = session.lock().await;

        match handler.handle_model_loaded(loaded).await {
            Ok(()) => Ok(Response::new(proto::ModelLoadedResponse {
                ok: Some(proto::Ok {}),
            })),
            Err(msg) => Err(Status::failed_precondition(msg)),
        }
    }

    async fn model_unloaded(
        &self,
        request: Request<proto::ModelUnloadedRequest>,
    ) -> Result<Response<proto::ModelUnloadedResponse>, Status> {
        let req = request.into_inner();
        if req.pod_name.is_empty() {
            return Err(Status::invalid_argument("pod_name is required"));
        }
        let unloaded = req
            .unloaded
            .ok_or_else(|| Status::invalid_argument("unloaded field is required"))?;

        let session = self.get_or_create_session(&req.pod_name, "").await;
        let mut handler = session.lock().await;

        match handler.handle_model_unloaded(unloaded).await {
            Ok(()) => Ok(Response::new(proto::ModelUnloadedResponse {
                ok: Some(proto::Ok {}),
            })),
            Err(msg) => Err(Status::failed_precondition(msg)),
        }
    }
}
