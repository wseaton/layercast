//! Per-session state machine:
//!
//!   ┌───────────┐  PrepareModel   ┌───────────┐  ModelLoaded  ┌───────┐
//!   │ Connected ├────────────────►│ Preparing ├──────────────►│ Ready │
//!   └───────────┘                 └───────────┘               └───┬───┘
//!        ▲                                                        │
//!        │                    ModelUnloaded                        │
//!        └────────────────────────────────────────────────────────┘
//!
//!   Ready + disconnect → auto-cleanup
//!   Ready + PrepareModel → unadvertise old, transition to Preparing

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use prost::Message;
use tracing::{debug, info, warn};

use crate::proto;
use crate::state::ServerState;
use crate::transfer_plan;

enum SessionState {
    Connected,
    Preparing { model_id: String, tp_rank: u32 },
    Ready { agent_name: String },
}

impl SessionState {
    fn name(&self) -> &'static str {
        match self {
            SessionState::Connected => "Connected",
            SessionState::Preparing { .. } => "Preparing",
            SessionState::Ready { .. } => "Ready",
        }
    }
}

/// Poll interval when waiting for NIXL peers to register.
const PEER_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub struct SessionHandler {
    state: SessionState,
    server_state: Arc<ServerState>,
    pod_name: String,
    pod_ip: String,
    hf_api: hf_hub::api::tokio::Api,
    peer_discovery_timeout: Duration,
}

impl SessionHandler {
    pub fn new(
        server_state: Arc<ServerState>,
        pod_name: String,
        pod_ip: String,
        hf_api: hf_hub::api::tokio::Api,
        peer_discovery_timeout: Duration,
    ) -> Self {
        Self {
            state: SessionState::Connected,
            server_state,
            pod_name,
            pod_ip,
            hf_api,
            peer_discovery_timeout,
        }
    }

    pub async fn handle_prepare(
        &mut self,
        prepare: proto::PrepareModel,
        timeout_override: Option<Duration>,
    ) -> Result<proto::Prepared, String> {
        match &self.state {
            SessionState::Connected => {}
            SessionState::Ready { agent_name } => {
                info!(agent_name, "hot-swap: unadvertising previous model");
                self.server_state.unregister_agent(agent_name).await;
            }
            SessionState::Preparing { .. } => {
                return Err(format!(
                    "invalid message PrepareModel in state {}",
                    self.state.name()
                ));
            }
        }

        let model_id = prepare.model_id;
        let revision = if prepare.revision.is_empty() {
            "main".to_string()
        } else {
            prepare.revision
        };
        let tp_rank = prepare.tp_rank;

        info!(model_id, revision, tp_rank, "PrepareModel: starting");

        let model_info = self
            .get_model_info(&model_id, &revision)
            .await
            .map_err(|e| format!("failed to list model files: {e}"))?;

        let timeout = timeout_override.unwrap_or(self.peer_discovery_timeout);
        let peers = self.discover_peers(&model_id, tp_rank, timeout).await;
        let transfer_plan = transfer_plan::compute_transfer_plan(&peers);

        info!(
            model_id,
            revision,
            tp_rank,
            file_count = model_info.files.len(),
            peer_count = peers.len(),
            weight_map_entries = model_info.weight_map.len(),
            transfer_plan_entries = transfer_plan.len(),
            "PrepareModel: complete"
        );

        self.state = SessionState::Preparing {
            model_id: model_id.clone(),
            tp_rank,
        };

        Ok(proto::Prepared {
            files: model_info.files,
            peers,
            weight_map: model_info.weight_map,
            transfer_plan,
        })
    }

    pub async fn handle_model_loaded(&mut self, loaded: proto::ModelLoaded) -> Result<(), String> {
        match &self.state {
            SessionState::Preparing {
                model_id: prepared_model,
                tp_rank: prepared_rank,
            } => {
                if loaded.model_id != *prepared_model || loaded.tp_rank != *prepared_rank {
                    warn!(
                        expected_model = %prepared_model,
                        actual_model = loaded.model_id,
                        expected_rank = prepared_rank,
                        actual_rank = loaded.tp_rank,
                        "ModelLoaded model/rank doesn't match PrepareModel"
                    );
                }
            }
            other => {
                return Err(format!(
                    "invalid message ModelLoaded in state {}",
                    other.name()
                ));
            }
        }

        let agent_name = loaded.agent_name.clone();

        let peer_md = proto::PeerNixlMd {
            agent_name: agent_name.clone(),
            nixl_md: loaded.nixl_md,
            tensors: loaded.tensors,
        };
        let stored_bytes = Bytes::from(peer_md.encode_to_vec());

        self.server_state
            .register_peer(
                &agent_name,
                &loaded.model_id,
                loaded.tp_rank,
                &self.pod_name,
                &self.pod_ip,
                stored_bytes,
            )
            .await;

        self.server_state
            .track_pod_agent(&self.pod_name, &agent_name)
            .await;

        info!(
            agent_name,
            model_id = loaded.model_id,
            tp_rank = loaded.tp_rank,
            "ModelLoaded: metadata stored, peer registered"
        );

        self.state = SessionState::Ready { agent_name };
        Ok(())
    }

    pub async fn handle_model_unloaded(
        &mut self,
        unloaded: proto::ModelUnloaded,
    ) -> Result<(), String> {
        match &self.state {
            SessionState::Ready {
                agent_name: current_agent,
            } => {
                if unloaded.agent_name != *current_agent {
                    warn!(
                        requested = unloaded.agent_name,
                        current = current_agent,
                        "ModelUnloaded agent doesn't match current, cleaning up both"
                    );
                    self.server_state.unregister_agent(current_agent).await;
                }
            }
            other => {
                return Err(format!(
                    "invalid message ModelUnloaded in state {}",
                    other.name()
                ));
            }
        }

        self.server_state
            .unregister_agent(&unloaded.agent_name)
            .await;

        info!(
            agent_name = unloaded.agent_name,
            "model unloaded, session back to Connected"
        );

        self.state = SessionState::Connected;
        Ok(())
    }

    pub async fn cleanup_on_disconnect(&mut self) {
        if let SessionState::Ready { ref agent_name } = self.state {
            warn!(
                agent_name,
                pod = self.pod_name,
                "session disconnected without ModelUnloaded, auto-cleaning up"
            );
            self.server_state.unregister_agent(agent_name).await;
        }
        self.state = SessionState::Connected;
    }

    async fn get_model_info(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<crate::state::CachedModelInfo, hf_hub::api::tokio::ApiError> {
        if let Some(cached) = self
            .server_state
            .file_list_cache
            .get(model_id, revision)
            .await
        {
            debug!(
                model_id,
                revision,
                files = cached.files.len(),
                weight_map = cached.weight_map.len(),
                "model info cache hit"
            );
            return Ok(cached);
        }

        let files = self.fetch_model_files_from_hf(model_id, revision).await?;
        let weight_map = self.fetch_weight_map(model_id, revision).await;
        let info = crate::state::CachedModelInfo { files, weight_map };
        self.server_state
            .file_list_cache
            .insert(model_id, revision, info.clone())
            .await;
        debug!(
            model_id,
            revision,
            files = info.files.len(),
            weight_map = info.weight_map.len(),
            "model info fetched from HF API and cached"
        );
        Ok(info)
    }

    async fn fetch_model_files_from_hf(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<Vec<String>, hf_hub::api::tokio::ApiError> {
        let repo = self.hf_api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        let info = repo.info().await?;
        let files = info
            .siblings
            .iter()
            .map(|s| &s.rfilename)
            .filter(|f| f.ends_with(".safetensors"))
            .cloned()
            .collect();
        Ok(files)
    }

    async fn fetch_weight_map(&self, model_id: &str, revision: &str) -> HashMap<String, String> {
        let repo = self.hf_api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        let index_path = match repo.download("model.safetensors.index.json").await {
            Ok(path) => path,
            Err(e) => {
                debug!(model_id, error = %e, "no safetensors index (single-shard model?)");
                return HashMap::new();
            }
        };

        let contents = match tokio::fs::read_to_string(&index_path).await {
            Ok(c) => c,
            Err(e) => {
                warn!(model_id, error = %e, "failed to read safetensors index");
                return HashMap::new();
            }
        };

        #[derive(serde::Deserialize)]
        struct SafetensorsIndex {
            weight_map: HashMap<String, String>,
        }

        match serde_json::from_str::<SafetensorsIndex>(&contents) {
            Ok(index) => {
                debug!(
                    model_id,
                    entries = index.weight_map.len(),
                    "parsed safetensors index weight_map"
                );
                index.weight_map
            }
            Err(e) => {
                warn!(model_id, error = %e, "failed to parse safetensors index JSON");
                HashMap::new()
            }
        }
    }

    async fn discover_peers(
        &self,
        model_id: &str,
        tp_rank: u32,
        timeout: Duration,
    ) -> Vec<proto::PeerNixlMd> {
        let start = tokio::time::Instant::now();

        let entries = loop {
            let found = self.server_state.find_peers(model_id, tp_rank).await;
            if !found.is_empty() {
                break found;
            }
            if timeout.is_zero() || start.elapsed() >= timeout {
                info!(
                    model_id,
                    tp_rank,
                    elapsed_s = start.elapsed().as_secs(),
                    "no peers found, giving up"
                );
                return Vec::new();
            }
            info!(
                model_id,
                tp_rank,
                elapsed_s = start.elapsed().as_secs(),
                timeout_s = timeout.as_secs(),
                "no peers found yet, retrying in 5s"
            );
            tokio::time::sleep(PEER_POLL_INTERVAL).await;
        };

        let mut peers = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(md_bytes) = self.server_state.get_nixl_metadata(&entry.agent_name).await {
                match proto::PeerNixlMd::decode(md_bytes.as_ref()) {
                    Ok(peer_md) => peers.push(peer_md),
                    Err(e) => {
                        warn!(
                            agent = entry.agent_name,
                            error = %e,
                            "failed to decode PeerNixlMd from store"
                        );
                    }
                }
            } else {
                debug!(
                    agent = entry.agent_name,
                    "peer registered but NIXL metadata not available (post-failover?)"
                );
            }
        }

        peers
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::proto;
    use crate::session::SessionHandler;
    use crate::state::{CachedModelInfo, ServerState};

    fn test_handler(state: Arc<ServerState>) -> SessionHandler {
        SessionHandler::new(
            state,
            "test-pod".to_string(),
            "10.0.0.1".to_string(),
            hf_hub::api::tokio::Api::new().unwrap(),
            Duration::ZERO,
        )
    }

    #[tokio::test]
    async fn happy_path_prepare_loaded_unloaded() {
        let state = Arc::new(ServerState::new());
        state
            .file_list_cache
            .insert(
                "test/model",
                "main",
                CachedModelInfo::from_files(vec!["shard.safetensors".into()]),
            )
            .await;

        let mut handler = test_handler(Arc::clone(&state));

        // PrepareModel
        let prepared = handler
            .handle_prepare(
                proto::PrepareModel {
                    model_id: "test/model".into(),
                    revision: "main".into(),
                    tp_rank: 0,
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(prepared.files, vec!["shard.safetensors"]);
        assert!(prepared.peers.is_empty());

        // ModelLoaded
        handler
            .handle_model_loaded(proto::ModelLoaded {
                agent_name: "worker-0".into(),
                nixl_md: vec![0xCA, 0xFE],
                tensors: vec![],
                model_id: "test/model".into(),
                files: vec!["shard.safetensors".into()],
                tp_rank: 0,
            })
            .await
            .unwrap();

        // Verify metadata is stored
        assert!(state.get_nixl_metadata("worker-0").await.is_some());

        // ModelUnloaded
        handler
            .handle_model_unloaded(proto::ModelUnloaded {
                agent_name: "worker-0".into(),
            })
            .await
            .unwrap();

        assert!(state.get_nixl_metadata("worker-0").await.is_none());
    }

    #[tokio::test]
    async fn model_loaded_in_wrong_state_returns_error() {
        let state = Arc::new(ServerState::new());
        let mut handler = test_handler(state);

        let result = handler
            .handle_model_loaded(proto::ModelLoaded {
                agent_name: "x".into(),
                nixl_md: vec![],
                tensors: vec![],
                model_id: "m".into(),
                files: vec![],
                tp_rank: 0,
            })
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Connected"));
    }

    #[tokio::test]
    async fn model_unloaded_in_wrong_state_returns_error() {
        let state = Arc::new(ServerState::new());
        let mut handler = test_handler(state);

        let result = handler
            .handle_model_unloaded(proto::ModelUnloaded {
                agent_name: "x".into(),
            })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn hot_swap_unadvertises_old() {
        let state = Arc::new(ServerState::new());
        state
            .file_list_cache
            .insert(
                "model-a",
                "main",
                CachedModelInfo::from_files(vec!["a.safetensors".into()]),
            )
            .await;
        state
            .file_list_cache
            .insert(
                "model-b",
                "main",
                CachedModelInfo::from_files(vec!["b.safetensors".into()]),
            )
            .await;

        let mut handler = test_handler(Arc::clone(&state));

        // Load model A
        handler
            .handle_prepare(
                proto::PrepareModel {
                    model_id: "model-a".into(),
                    revision: "main".into(),
                    tp_rank: 0,
                },
                None,
            )
            .await
            .unwrap();
        handler
            .handle_model_loaded(proto::ModelLoaded {
                agent_name: "agent-a".into(),
                nixl_md: vec![0xAA],
                tensors: vec![],
                model_id: "model-a".into(),
                files: vec!["a.safetensors".into()],
                tp_rank: 0,
            })
            .await
            .unwrap();

        assert!(state.get_nixl_metadata("agent-a").await.is_some());

        // Hot-swap to model B
        handler
            .handle_prepare(
                proto::PrepareModel {
                    model_id: "model-b".into(),
                    revision: "main".into(),
                    tp_rank: 0,
                },
                None,
            )
            .await
            .unwrap();

        // Model A should be cleaned up
        assert!(state.get_nixl_metadata("agent-a").await.is_none());
    }

    #[tokio::test]
    async fn cleanup_on_disconnect_in_ready_state() {
        let state = Arc::new(ServerState::new());
        state
            .file_list_cache
            .insert(
                "test/model",
                "main",
                CachedModelInfo::from_files(vec!["s.safetensors".into()]),
            )
            .await;

        let mut handler = test_handler(Arc::clone(&state));

        handler
            .handle_prepare(
                proto::PrepareModel {
                    model_id: "test/model".into(),
                    revision: "main".into(),
                    tp_rank: 0,
                },
                None,
            )
            .await
            .unwrap();
        handler
            .handle_model_loaded(proto::ModelLoaded {
                agent_name: "crash-test".into(),
                nixl_md: vec![0xFF],
                tensors: vec![],
                model_id: "test/model".into(),
                files: vec!["s.safetensors".into()],
                tp_rank: 0,
            })
            .await
            .unwrap();

        assert!(state.get_nixl_metadata("crash-test").await.is_some());

        handler.cleanup_on_disconnect().await;

        assert!(state.get_nixl_metadata("crash-test").await.is_none());
    }

    #[tokio::test]
    async fn discover_peers_from_global_state() {
        let state = Arc::new(ServerState::new());
        state
            .file_list_cache
            .insert(
                "test/model",
                "main",
                CachedModelInfo::from_files(vec!["s.safetensors".into()]),
            )
            .await;

        // Register a peer in the global state (as if another pod loaded it)
        let peer_md = proto::PeerNixlMd {
            agent_name: "seed-0-rank0".to_string(),
            nixl_md: vec![0xDE, 0xAD],
            tensors: vec![proto::TensorInfo {
                name: "layer.0.weight".into(),
                size: 4096,
                ..Default::default()
            }],
        };
        use prost::Message;
        state
            .register_peer(
                "seed-0-rank0",
                "test/model",
                0,
                "seed-0",
                "10.0.0.42",
                bytes::Bytes::from(peer_md.encode_to_vec()),
            )
            .await;

        // A different pod's handler should discover the peer
        let mut handler = test_handler(Arc::clone(&state));
        let prepared = handler
            .handle_prepare(
                proto::PrepareModel {
                    model_id: "test/model".into(),
                    revision: "main".into(),
                    tp_rank: 0,
                },
                None,
            )
            .await
            .unwrap();

        assert_eq!(prepared.peers.len(), 1);
        assert_eq!(prepared.peers[0].agent_name, "seed-0-rank0");
        assert_eq!(prepared.peers[0].tensors.len(), 1);
    }
}
