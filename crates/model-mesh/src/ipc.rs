use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use discovery::PeerDiscovery;

use crate::nixl_vram_store::ModelPeerStore;

/// Errors that can occur in the IPC server and its supporting operations.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to serialize message: {0}")]
    Serialize(#[from] rmp_serde::encode::Error),

    #[error("failed to deserialize message: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),

    #[error("message too large: {size} bytes (max {max})")]
    MessageTooLarge { size: u32, max: u32 },

    #[error("HF Hub API error: {0}")]
    HfHub(#[from] hf_hub::api::tokio::ApiError),

    #[error("IPC server task failed: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
}

// Wire protocol messages
//
// Three request types form a state machine per connection:
//
//   Connected ──PrepareModel──► Preparing
//   Preparing ──ModelLoaded───► Ready
//   Ready ─────ModelUnloaded──► Connected
//   Ready ─────PrepareModel───► Preparing  (hot-swap)
//   Ready ─────EOF────────────► cleanup    (crash protection)

/// Maximum allowed message size (64 MB).
const MAX_MESSAGE_SIZE: u32 = 64 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcRequest {
    /// Batch: list model files + discover NIXL peers + fetch metadata.
    /// Valid in: Connected, Ready (triggers hot-swap unadvertise).
    #[serde(rename = "prepare_model")]
    PrepareModel {
        model_id: String,
        revision: String,
        tp_rank: u32,
    },

    /// Batch: store NIXL metadata + advertise via CRD.
    /// Valid in: Preparing only.
    #[serde(rename = "model_loaded")]
    ModelLoaded {
        agent_name: String,
        #[serde(with = "serde_bytes")]
        metadata: Vec<u8>,
        model_id: String,
        files: Vec<String>,
        tp_rank: u32,
    },

    /// Explicit teardown: unadvertise + remove metadata.
    /// Valid in: Ready only.
    #[serde(rename = "model_unloaded")]
    ModelUnloaded { agent_name: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcResponse {
    /// Response to PrepareModel: safetensor file list + peer NIXL metadata.
    #[serde(rename = "prepared")]
    Prepared {
        files: Vec<String>,
        peers: Vec<PeerNixlMdInfo>,
    },

    #[serde(rename = "ok")]
    Ok,

    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PeerNixlMdInfo {
    pub agent_name: String,
    #[serde(with = "serde_bytes")]
    pub metadata: Vec<u8>,
}

// Per-connection state machine

/// Tracks what the client is doing so we can enforce valid transitions
/// and auto-cleanup on crash (EOF while Ready).
enum ConnectionState {
    /// Fresh connection or after ModelUnloaded. Accepts: PrepareModel.
    Connected,
    /// PrepareModel received, waiting for ModelLoaded. Accepts: ModelLoaded.
    Preparing { model_id: String, tp_rank: u32 },
    /// Model loaded and advertised. Accepts: ModelUnloaded, PrepareModel (hot-swap).
    /// EOF here triggers auto-unadvertise.
    Ready { agent_name: String },
}

impl ConnectionState {
    fn name(&self) -> &'static str {
        match self {
            ConnectionState::Connected => "Connected",
            ConnectionState::Preparing { .. } => "Preparing",
            ConnectionState::Ready { .. } => "Ready",
        }
    }
}

// Length-prefixed msgpack helpers

async fn read_message<T: DeserializeOwned>(stream: &mut UnixStream) -> Result<T, IpcError> {
    let len = stream.read_u32().await?;
    if len > MAX_MESSAGE_SIZE {
        return Err(IpcError::MessageTooLarge {
            size: len,
            max: MAX_MESSAGE_SIZE,
        });
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    Ok(rmp_serde::from_slice(&buf)?)
}

async fn write_message<T: Serialize>(stream: &mut UnixStream, msg: &T) -> Result<(), IpcError> {
    let data = rmp_serde::to_vec_named(msg)?;
    stream.write_u32(data.len() as u32).await?;
    stream.write_all(&data).await?;
    stream.flush().await?;
    Ok(())
}

// File list cache (populated by predictive prefetch)

/// Pre-fetched file lists, keyed by "model_id@revision".
/// Populated at startup if MODEL_NAME is set, otherwise filled lazily.
#[derive(Clone, Default)]
pub struct FileListCache {
    inner: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl FileListCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn cache_key(model_id: &str, revision: &str) -> String {
        format!("{model_id}@{revision}")
    }

    pub async fn get(&self, model_id: &str, revision: &str) -> Option<Vec<String>> {
        self.inner
            .read()
            .await
            .get(&Self::cache_key(model_id, revision))
            .cloned()
    }

    pub async fn insert(&self, model_id: &str, revision: &str, files: Vec<String>) {
        self.inner
            .write()
            .await
            .insert(Self::cache_key(model_id, revision), files);
    }
}

// IPC server

/// Handle returned by `IpcServer::start` so callers can manage the accept loop.
pub struct ServeHandle {
    handle: tokio::task::JoinHandle<()>,
}

impl ServeHandle {
    pub async fn join(self) -> Result<(), IpcError> {
        self.handle.await?;
        Ok(())
    }

    pub fn abort(&self) {
        self.handle.abort();
    }
}

/// Unix socket IPC server for vLLM plugin communication.
///
/// Each connection is a state machine:
///
///   ┌───────────┐  PrepareModel   ┌───────────┐  ModelLoaded  ┌───────┐
///   │ Connected ├────────────────►│ Preparing ├──────────────►│ Ready │
///   └───────────┘                 └───────────┘               └───┬───┘
///        ▲                                                        │
///        │                    ModelUnloaded                        │
///        └────────────────────────────────────────────────────────┘
///
///   Ready + EOF  →  auto-unadvertise (crash protection)
///   Ready + PrepareModel  →  unadvertise old, transition to Preparing
/// How often to poll for peers when `peer_discovery_timeout` > 0.
const PEER_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub struct IpcServer {
    socket_path: PathBuf,
    discovery: Arc<dyn PeerDiscovery>,
    hf_api: hf_hub::api::tokio::Api,
    http_client: reqwest::Client,
    model_peer_store: ModelPeerStore,
    http_port: u16,
    file_cache: FileListCache,
    peer_discovery_timeout: Duration,
}

impl IpcServer {
    pub fn new(
        socket_path: impl Into<PathBuf>,
        discovery: Arc<dyn PeerDiscovery>,
        hf_api: hf_hub::api::tokio::Api,
        model_peer_store: ModelPeerStore,
        http_port: u16,
        file_cache: FileListCache,
        peer_discovery_timeout: Duration,
    ) -> Self {
        Self {
            socket_path: socket_path.into(),
            discovery,
            hf_api,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            model_peer_store,
            http_port,
            file_cache,
            peer_discovery_timeout,
        }
    }

    pub fn start(self: Arc<Self>) -> Result<ServeHandle, IpcError> {
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = UnixListener::bind(&self.socket_path)?;

        info!(path = %self.socket_path.display(), "IPC server listening");

        let server = Arc::clone(&self);
        let handle = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        #[cfg(target_os = "linux")]
                        if let Ok(cred) = stream.peer_cred() {
                            debug!(
                                pid = cred.pid().unwrap_or(0),
                                uid = cred.uid(),
                                "IPC client connected"
                            );
                        }

                        let srv = Arc::clone(&server);
                        tokio::spawn(async move {
                            if let Err(e) = srv.handle_connection(stream).await {
                                warn!(error = %e, "IPC connection handler error");
                            }
                        });
                    }
                    Err(e) => {
                        error!(error = %e, "IPC accept error");
                        break;
                    }
                }
            }
        });

        Ok(ServeHandle { handle })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// File cache reference, so main.rs can pre-populate it.
    pub fn file_cache(&self) -> &FileListCache {
        &self.file_cache
    }

    async fn handle_connection(&self, mut stream: UnixStream) -> Result<(), IpcError> {
        let mut state = ConnectionState::Connected;

        loop {
            let request: IpcRequest = match read_message(&mut stream).await {
                Ok(req) => req,
                Err(e) => {
                    let is_eof = matches!(
                        e,
                        IpcError::Io(ref io) if io.kind() == std::io::ErrorKind::UnexpectedEof
                    );

                    if is_eof {
                        // Crash protection: if we were in Ready state,
                        // the plugin died without sending ModelUnloaded.
                        // Unadvertise immediately so stale peers don't
                        // try to RDMA from a dead process.
                        if let ConnectionState::Ready { ref agent_name } = state {
                            warn!(
                                agent_name,
                                "plugin disconnected without ModelUnloaded, auto-unadvertising"
                            );
                            self.cleanup_agent(agent_name).await;
                        }
                    } else {
                        warn!(error = %e, "IPC read error");
                        if let ConnectionState::Ready { ref agent_name } = state {
                            self.cleanup_agent(agent_name).await;
                        }
                    }
                    break;
                }
            };

            debug!(state = state.name(), ?request, "IPC request received");

            let (response, next_state) = self.dispatch(request, state).await;
            state = next_state;

            write_message(&mut stream, &response).await?;
        }
        Ok(())
    }

    /// Route a request through the state machine. Returns the response and
    /// the next state. Invalid transitions return an error response and
    /// leave the state unchanged.
    async fn dispatch(
        &self,
        request: IpcRequest,
        state: ConnectionState,
    ) -> (IpcResponse, ConnectionState) {
        match (request, state) {
            // ── Connected + PrepareModel → Preparing ──────────────────
            (
                IpcRequest::PrepareModel {
                    model_id,
                    revision,
                    tp_rank,
                },
                ConnectionState::Connected,
            ) => self.handle_prepare_model(model_id, revision, tp_rank).await,

            // ── Ready + PrepareModel → hot-swap (unadvertise old, prepare new)
            (
                IpcRequest::PrepareModel {
                    model_id,
                    revision,
                    tp_rank,
                },
                ConnectionState::Ready { agent_name },
            ) => {
                info!(agent_name, "hot-swap: unadvertising previous model");
                self.cleanup_agent(&agent_name).await;
                self.handle_prepare_model(model_id, revision, tp_rank).await
            }

            // ── Preparing + ModelLoaded → Ready ──────────────────────
            (
                IpcRequest::ModelLoaded {
                    agent_name,
                    metadata,
                    model_id,
                    files,
                    tp_rank,
                },
                ConnectionState::Preparing {
                    model_id: prepared_model,
                    tp_rank: prepared_rank,
                },
            ) => {
                // Sanity check: the model_id and tp_rank should match what
                // was prepared. Not a hard error (the client knows best),
                // but worth flagging.
                if model_id != prepared_model || tp_rank != prepared_rank {
                    warn!(
                        expected_model = prepared_model,
                        actual_model = model_id,
                        expected_rank = prepared_rank,
                        actual_rank = tp_rank,
                        "ModelLoaded model/rank doesn't match PrepareModel"
                    );
                }

                self.handle_model_loaded(agent_name, metadata, model_id, files, tp_rank)
                    .await
            }

            // ── Ready + ModelUnloaded → Connected ────────────────────
            (
                IpcRequest::ModelUnloaded { agent_name },
                ConnectionState::Ready {
                    agent_name: current_agent,
                },
            ) => {
                if agent_name != current_agent {
                    warn!(
                        requested = agent_name,
                        current = current_agent,
                        "ModelUnloaded agent doesn't match current, cleaning up both"
                    );
                    self.cleanup_agent(&current_agent).await;
                }
                self.cleanup_agent(&agent_name).await;
                info!(agent_name, "model unloaded, connection back to Connected");
                (IpcResponse::Ok, ConnectionState::Connected)
            }

            // ── Invalid transitions ──────────────────────────────────
            (request, state) => {
                let msg = format!(
                    "invalid message {:?} in state {}",
                    std::mem::discriminant(&request),
                    state.name(),
                );
                warn!(msg, "IPC state machine violation");
                (IpcResponse::Error { message: msg }, state)
            }
        }
    }

    async fn handle_prepare_model(
        &self,
        model_id: String,
        revision: String,
        tp_rank: u32,
    ) -> (IpcResponse, ConnectionState) {
        info!(model_id, revision, tp_rank, "PrepareModel: starting");

        // Step 1: Get file list (from cache or HF API).
        let files = match self.get_model_files(&model_id, &revision).await {
            Ok(f) => f,
            Err(e) => {
                return (
                    IpcResponse::Error {
                        message: format!("failed to list model files: {e}"),
                    },
                    ConnectionState::Connected,
                );
            }
        };

        // Step 2: Find model peers and fetch their metadata.
        let peers = self.discover_and_fetch_peers(&model_id, tp_rank).await;

        info!(
            model_id,
            revision,
            tp_rank,
            file_count = files.len(),
            peer_count = peers.len(),
            "PrepareModel: complete"
        );

        let next_state = ConnectionState::Preparing { model_id, tp_rank };

        (IpcResponse::Prepared { files, peers }, next_state)
    }

    async fn handle_model_loaded(
        &self,
        agent_name: String,
        metadata: Vec<u8>,
        model_id: String,
        files: Vec<String>,
        tp_rank: u32,
    ) -> (IpcResponse, ConnectionState) {
        // Store raw NIXL metadata locally (served to peers via HTTP)
        self.model_peer_store
            .insert(&agent_name, metadata.to_vec())
            .await;

        // Advertise lightweight pointer via discovery CRD
        self.discovery
            .advertise_model_peer(&agent_name, &model_id, tp_rank)
            .await;

        info!(
            agent_name,
            model_id,
            tp_rank,
            metadata_len = metadata.len(),
            file_count = files.len(),
            "ModelLoaded: metadata stored, CRD advertised"
        );

        let next_state = ConnectionState::Ready {
            agent_name: agent_name.clone(),
        };
        (IpcResponse::Ok, next_state)
    }

    /// Remove all traces of an agent: unadvertise from CRD, remove from store.
    async fn cleanup_agent(&self, agent_name: &str) {
        self.discovery.unadvertise_model_peer(agent_name).await;
        self.model_peer_store.remove(agent_name).await;
        info!(
            agent_name,
            "agent cleaned up (unadvertised + metadata removed)"
        );
    }

    /// Get safetensor files for a model, checking the prefetch cache first.
    async fn get_model_files(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<Vec<String>, IpcError> {
        // Check cache first (populated by predictive prefetch or previous PrepareModel)
        if let Some(cached) = self.file_cache.get(model_id, revision).await {
            debug!(
                model_id,
                revision,
                count = cached.len(),
                "file list cache hit"
            );
            return Ok(cached);
        }

        // Cache miss: fetch from HF API
        let files = self.fetch_model_files_from_hf(model_id, revision).await?;
        self.file_cache
            .insert(model_id, revision, files.clone())
            .await;
        debug!(
            model_id,
            revision,
            count = files.len(),
            "file list fetched from HF API and cached"
        );
        Ok(files)
    }

    async fn fetch_model_files_from_hf(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<Vec<String>, IpcError> {
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

    /// Discover model peers for a given model + TP rank and fetch their metadata.
    ///
    /// If no peers are found on the first attempt and `peer_discovery_timeout` is
    /// non-zero, polls every 5s until peers appear or the timeout expires. This
    /// covers the window where a seed's CRD advertisement hasn't propagated through
    /// the K8s reflector yet.
    async fn discover_and_fetch_peers(&self, model_id: &str, tp_rank: u32) -> Vec<PeerNixlMdInfo> {
        let start = tokio::time::Instant::now();
        let peers = loop {
            let found = self.discovery.find_model_peers(model_id, tp_rank).await;
            if !found.is_empty() {
                break found;
            }
            if self.peer_discovery_timeout.is_zero()
                || start.elapsed() >= self.peer_discovery_timeout
            {
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
                timeout_s = self.peer_discovery_timeout.as_secs(),
                "no peers found yet, retrying in 5s"
            );
            tokio::time::sleep(PEER_POLL_INTERVAL).await;
        };

        // Fetch metadata from all peers concurrently
        let http_port = self.http_port;
        let futs: Vec<_> = peers
            .into_iter()
            .map(|peer| {
                let agent_name = peer.agent_name;
                let peer_ip = peer.peer_addr.ip();
                let client = &self.http_client;
                let store = &self.model_peer_store;

                async move {
                    // Try local store first (only has metadata for agents
                    // loaded by THIS pod, not other pods on the same node).
                    if let Some(metadata) = store.get(&agent_name).await {
                        debug!(agent = %agent_name, len = metadata.len(), "local NIXL metadata (own store)");
                        return Some(PeerNixlMdInfo {
                            agent_name,
                            metadata: metadata.to_vec(),
                        });
                    }

                    // Fetch via HTTP (works for both same-node and cross-node peers)
                    let url = format!(
                        "http://{}:{}/internal/nixl-vram/{}",
                        peer_ip, http_port, agent_name
                    );
                    debug!(agent = %agent_name, %url, "fetching peer NIXL metadata");

                    match client.get(&url).send().await {
                        Ok(resp) if resp.status().is_success() => match resp.bytes().await {
                            Ok(bytes) => {
                                info!(agent = %agent_name, len = bytes.len(), "fetched peer NIXL metadata");
                                Some(PeerNixlMdInfo {
                                    agent_name,
                                    metadata: bytes.to_vec(),
                                })
                            }
                            Err(e) => {
                                warn!(agent = %agent_name, error = %e, "failed to read peer metadata body");
                                None
                            }
                        },
                        Ok(resp) => {
                            warn!(agent = %agent_name, status = %resp.status(), "peer returned non-success");
                            None
                        }
                        Err(e) => {
                            warn!(agent = %agent_name, %url, error = %e, "failed to fetch peer metadata");
                            None
                        }
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(futs).await;
        results.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::ipc::{
        FileListCache, IpcRequest, IpcResponse, IpcServer, PeerNixlMdInfo, read_message,
        write_message,
    };
    use crate::nixl_vram_store::ModelPeerStore;
    use std::sync::Arc;
    use std::time::Duration;

    use discovery::{MockDiscovery, PeerDiscovery};
    use tokio::net::UnixStream;

    async fn test_server(dir: &std::path::Path) -> Arc<IpcServer> {
        let discovery: Arc<dyn discovery::PeerDiscovery> =
            Arc::new(MockDiscovery::new("test-node"));

        let socket_path = dir.join("layercast.sock");
        Arc::new(IpcServer::new(
            socket_path,
            discovery,
            hf_hub::api::tokio::Api::new().unwrap(),
            ModelPeerStore::new(),
            8081,
            FileListCache::new(),
            Duration::ZERO, // no retry in tests
        ))
    }

    // Serialization round-trips

    #[test]
    fn request_roundtrip_prepare_model() {
        let req = IpcRequest::PrepareModel {
            model_id: "meta-llama/70b".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: IpcRequest = rmp_serde::from_slice(&bytes).unwrap();

        match decoded {
            IpcRequest::PrepareModel {
                model_id,
                revision,
                tp_rank,
            } => {
                assert_eq!(model_id, "meta-llama/70b");
                assert_eq!(revision, "main");
                assert_eq!(tp_rank, 0);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn request_roundtrip_model_loaded() {
        let req = IpcRequest::ModelLoaded {
            agent_name: "worker-0".into(),
            metadata: vec![0xCA, 0xFE],
            model_id: "m".into(),
            files: vec!["f.safetensors".into()],
            tp_rank: 1,
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: IpcRequest = rmp_serde::from_slice(&bytes).unwrap();

        match decoded {
            IpcRequest::ModelLoaded {
                agent_name,
                metadata,
                tp_rank,
                ..
            } => {
                assert_eq!(agent_name, "worker-0");
                assert_eq!(metadata, vec![0xCA, 0xFE]);
                assert_eq!(tp_rank, 1);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn request_roundtrip_model_unloaded() {
        let req = IpcRequest::ModelUnloaded {
            agent_name: "worker-0".into(),
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: IpcRequest = rmp_serde::from_slice(&bytes).unwrap();

        match decoded {
            IpcRequest::ModelUnloaded { agent_name } => {
                assert_eq!(agent_name, "worker-0");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn response_roundtrip_prepared() {
        let resp = IpcResponse::Prepared {
            files: vec!["shard-001.safetensors".into()],
            peers: vec![PeerNixlMdInfo {
                agent_name: "peer-0".into(),
                metadata: vec![0xDE, 0xAD],
            }],
        };
        let bytes = rmp_serde::to_vec_named(&resp).unwrap();
        let decoded: IpcResponse = rmp_serde::from_slice(&bytes).unwrap();

        match decoded {
            IpcResponse::Prepared { files, peers } => {
                assert_eq!(files, vec!["shard-001.safetensors"]);
                assert_eq!(peers.len(), 1);
                assert_eq!(peers[0].agent_name, "peer-0");
                assert_eq!(peers[0].metadata, vec![0xDE, 0xAD]);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn all_request_variants_roundtrip() {
        let requests: Vec<IpcRequest> = vec![
            IpcRequest::PrepareModel {
                model_id: "m".into(),
                revision: "main".into(),
                tp_rank: 0,
            },
            IpcRequest::ModelLoaded {
                agent_name: "a".into(),
                metadata: vec![1],
                model_id: "m".into(),
                files: vec!["f".into()],
                tp_rank: 0,
            },
            IpcRequest::ModelUnloaded {
                agent_name: "a".into(),
            },
        ];

        for req in &requests {
            let bytes = rmp_serde::to_vec_named(req).unwrap();
            let _decoded: IpcRequest = rmp_serde::from_slice(&bytes).unwrap();
        }
    }

    #[test]
    fn all_response_variants_roundtrip() {
        let responses: Vec<IpcResponse> = vec![
            IpcResponse::Prepared {
                files: vec![],
                peers: vec![],
            },
            IpcResponse::Ok,
            IpcResponse::Error {
                message: "boom".into(),
            },
        ];

        for resp in &responses {
            let bytes = rmp_serde::to_vec_named(resp).unwrap();
            let _decoded: IpcResponse = rmp_serde::from_slice(&bytes).unwrap();
        }
    }

    // File list cache

    #[tokio::test]
    async fn file_cache_miss_and_hit() {
        let cache = FileListCache::new();
        assert!(cache.get("m", "main").await.is_none());

        cache
            .insert("m", "main", vec!["a.safetensors".into()])
            .await;
        let files = cache.get("m", "main").await.unwrap();
        assert_eq!(files, vec!["a.safetensors"]);

        // Different revision is a miss
        assert!(cache.get("m", "dev").await.is_none());
    }

    // State machine integration tests (real Unix sockets)

    #[tokio::test]
    async fn state_machine_happy_path() {
        // Connected → PrepareModel → Preparing → ModelLoaded → Ready → ModelUnloaded → Connected
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        // Pre-populate file cache so PrepareModel doesn't hit HF API
        server
            .file_cache()
            .insert("test/model", "main", vec!["shard-001.safetensors".into()])
            .await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        // PrepareModel
        let req = IpcRequest::PrepareModel {
            model_id: "test/model".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        match resp {
            IpcResponse::Prepared { files, peers } => {
                assert_eq!(files, vec!["shard-001.safetensors"]);
                assert!(peers.is_empty()); // no peers in mock
            }
            other => panic!("expected Prepared, got {other:?}"),
        }

        // ModelLoaded
        let req = IpcRequest::ModelLoaded {
            agent_name: "worker-0".into(),
            metadata: vec![0xCA, 0xFE],
            model_id: "test/model".into(),
            files: vec!["shard-001.safetensors".into()],
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        assert!(matches!(resp, IpcResponse::Ok));

        // ModelUnloaded
        let req = IpcRequest::ModelUnloaded {
            agent_name: "worker-0".into(),
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        assert!(matches!(resp, IpcResponse::Ok));

        handle.abort();
    }

    #[tokio::test]
    async fn state_machine_invalid_transition_returns_error() {
        // Connected + ModelLoaded should fail (need PrepareModel first)
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        let req = IpcRequest::ModelLoaded {
            agent_name: "worker-0".into(),
            metadata: vec![0xDE, 0xAD],
            model_id: "test/model".into(),
            files: vec!["f.safetensors".into()],
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();

        match resp {
            IpcResponse::Error { message } => {
                assert!(message.contains("invalid message"));
                assert!(message.contains("Connected"));
            }
            other => panic!("expected Error, got {other:?}"),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn state_machine_model_unloaded_in_wrong_state() {
        // Connected + ModelUnloaded should fail
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        let req = IpcRequest::ModelUnloaded {
            agent_name: "ghost".into(),
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        assert!(matches!(resp, IpcResponse::Error { .. }));

        handle.abort();
    }

    #[tokio::test]
    async fn eof_in_ready_state_auto_unadvertises() {
        // If the plugin crashes (EOF) while in Ready, the sidecar should
        // auto-unadvertise to prevent stale RDMA targets.
        let tmp = tempfile::tempdir().unwrap();
        let discovery = Arc::new(MockDiscovery::new("test-node"));
        let nixl_store = ModelPeerStore::new();
        let socket_path = tmp.path().join("layercast.sock");

        let server = Arc::new(IpcServer::new(
            socket_path.clone(),
            Arc::clone(&discovery) as Arc<dyn discovery::PeerDiscovery>,
            hf_hub::api::tokio::Api::new().unwrap(),
            nixl_store.clone(),
            8081,
            FileListCache::new(),
            Duration::ZERO,
        ));

        server
            .file_cache()
            .insert("test/model", "main", vec!["shard.safetensors".into()])
            .await;

        let handle = Arc::clone(&server).start().unwrap();
        {
            let mut stream = UnixStream::connect(&socket_path).await.unwrap();

            // PrepareModel
            let req = IpcRequest::PrepareModel {
                model_id: "test/model".into(),
                revision: "main".into(),
                tp_rank: 0,
            };
            write_message(&mut stream, &req).await.unwrap();
            let _: IpcResponse = read_message(&mut stream).await.unwrap();

            // ModelLoaded
            let req = IpcRequest::ModelLoaded {
                agent_name: "crash-test".into(),
                metadata: vec![0xFF],
                model_id: "test/model".into(),
                files: vec!["shard.safetensors".into()],
                tp_rank: 0,
            };
            write_message(&mut stream, &req).await.unwrap();
            let _: IpcResponse = read_message(&mut stream).await.unwrap();

            // Verify metadata is stored
            assert!(nixl_store.get("crash-test").await.is_some());

            // Drop the stream (simulating a crash / EOF)
        }

        // Give the server a moment to process the EOF cleanup
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Metadata should be removed by crash protection
        assert!(
            nixl_store.get("crash-test").await.is_none(),
            "metadata should be cleaned up after EOF in Ready state"
        );

        // Discovery should have been unadvertised too
        let peers = discovery.find_model_peers("test/model", 0).await;
        assert!(peers.is_empty(), "peers should be empty after EOF cleanup");

        handle.abort();
    }

    #[tokio::test]
    async fn hot_swap_unadvertises_old_model() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery = Arc::new(MockDiscovery::new("test-node"));
        let nixl_store = ModelPeerStore::new();
        let socket_path = tmp.path().join("layercast.sock");

        let server = Arc::new(IpcServer::new(
            socket_path.clone(),
            Arc::clone(&discovery) as Arc<dyn discovery::PeerDiscovery>,
            hf_hub::api::tokio::Api::new().unwrap(),
            nixl_store.clone(),
            8081,
            FileListCache::new(),
            Duration::ZERO,
        ));

        server
            .file_cache()
            .insert("model-a", "main", vec!["a.safetensors".into()])
            .await;
        server
            .file_cache()
            .insert("model-b", "main", vec!["b.safetensors".into()])
            .await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(&socket_path).await.unwrap();

        // Load model A
        let req = IpcRequest::PrepareModel {
            model_id: "model-a".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let _: IpcResponse = read_message(&mut stream).await.unwrap();

        let req = IpcRequest::ModelLoaded {
            agent_name: "agent-a".into(),
            metadata: vec![0xAA],
            model_id: "model-a".into(),
            files: vec!["a.safetensors".into()],
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let _: IpcResponse = read_message(&mut stream).await.unwrap();

        // Verify model A is advertised
        assert!(nixl_store.get("agent-a").await.is_some());

        // Hot-swap: PrepareModel for model B while in Ready
        let req = IpcRequest::PrepareModel {
            model_id: "model-b".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let _: IpcResponse = read_message(&mut stream).await.unwrap();

        // Model A should be cleaned up
        assert!(
            nixl_store.get("agent-a").await.is_none(),
            "old model metadata should be removed on hot-swap"
        );

        handle.abort();
    }

    #[tokio::test]
    async fn prepare_model_with_cached_files() {
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        server
            .file_cache()
            .insert(
                "org/model",
                "v1",
                vec![
                    "shard-001.safetensors".into(),
                    "shard-002.safetensors".into(),
                ],
            )
            .await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        let req = IpcRequest::PrepareModel {
            model_id: "org/model".into(),
            revision: "v1".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();

        match resp {
            IpcResponse::Prepared { files, .. } => {
                assert_eq!(files.len(), 2);
                assert!(files.contains(&"shard-001.safetensors".to_string()));
                assert!(files.contains(&"shard-002.safetensors".to_string()));
            }
            other => panic!("expected Prepared, got {other:?}"),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn prepare_model_hf_unreachable_returns_error() {
        // No file cache, HF API unreachable → error response, stays Connected
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        let handle = Arc::clone(&server).start().unwrap();
        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        let req = IpcRequest::PrepareModel {
            model_id: "org/model".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        assert!(matches!(resp, IpcResponse::Error { .. }));

        // Should still be in Connected state, so PrepareModel again should work
        // (once we have a cache hit)
        server
            .file_cache()
            .insert("org/model", "main", vec!["f.safetensors".into()])
            .await;
        let req = IpcRequest::PrepareModel {
            model_id: "org/model".into(),
            revision: "main".into(),
            tp_rank: 0,
        };
        write_message(&mut stream, &req).await.unwrap();
        let resp: IpcResponse = read_message(&mut stream).await.unwrap();
        assert!(matches!(resp, IpcResponse::Prepared { .. }));

        handle.abort();
    }

    #[tokio::test]
    async fn rejects_oversized_message() {
        let tmp = tempfile::tempdir().unwrap();
        let server = test_server(tmp.path()).await;

        let handle = Arc::clone(&server).start().unwrap();

        let mut stream = UnixStream::connect(server.socket_path()).await.unwrap();

        use tokio::io::AsyncWriteExt;
        let fake_len: u32 = 128 * 1024 * 1024;
        stream.write_u32(fake_len).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let result: Result<IpcResponse, _> = read_message(&mut stream).await;
        assert!(
            result.is_err(),
            "oversized message should cause connection drop"
        );

        handle.abort();
    }
}
