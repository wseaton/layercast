//!   ┌─────────────────────────────────────────────┐
//!   │              ServerState                     │
//!   │  ┌───────────────┐  ┌─────────────────────┐ │
//!   │  │ peer_registry │  │ nixl_metadata        │ │
//!   │  │ agent -> entry│  │ agent -> raw bytes   │ │
//!   │  └───────────────┘  └─────────────────────┘ │
//!   │  ┌───────────────┐  ┌─────────────────────┐ │
//!   │  │ pod_agents    │  │ file_list_cache      │ │
//!   │  │ pod -> agents │  │ model@rev -> info    │ │
//!   │  └───────────────┘  └─────────────────────┘ │
//!   └──────────────────────┬──────────────────────┘
//!                          │ debounced persist
//!                          ▼
//!                     ConfigMap

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPeerEntry {
    pub agent_name: String,
    pub model_id: String,
    pub tp_rank: u32,
    pub pod_name: String,
    pub pod_ip: String,
    pub registered_at: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CachedModelInfo {
    pub files: Vec<String>,
    pub weight_map: HashMap<String, String>,
}

impl CachedModelInfo {
    pub fn from_files(files: Vec<String>) -> Self {
        Self {
            files,
            weight_map: HashMap::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct FileListCache {
    inner: Arc<RwLock<HashMap<String, CachedModelInfo>>>,
}

impl FileListCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn cache_key(model_id: &str, revision: &str) -> String {
        format!("{model_id}@{revision}")
    }

    pub async fn get(&self, model_id: &str, revision: &str) -> Option<CachedModelInfo> {
        self.inner
            .read()
            .await
            .get(&Self::cache_key(model_id, revision))
            .cloned()
    }

    pub async fn insert(&self, model_id: &str, revision: &str, info: CachedModelInfo) {
        self.inner
            .write()
            .await
            .insert(Self::cache_key(model_id, revision), info);
    }
}

pub struct ServerState {
    pub peer_registry: RwLock<HashMap<String, ModelPeerEntry>>,
    pub nixl_metadata: RwLock<HashMap<String, Bytes>>,
    pub pod_agents: RwLock<HashMap<String, Vec<String>>>,
    pub file_list_cache: FileListCache,
    pub persist_notify: Notify,
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            peer_registry: RwLock::new(HashMap::new()),
            nixl_metadata: RwLock::new(HashMap::new()),
            pod_agents: RwLock::new(HashMap::new()),
            file_list_cache: FileListCache::new(),
            persist_notify: Notify::new(),
        }
    }

    pub async fn register_peer(
        &self,
        agent_name: &str,
        model_id: &str,
        tp_rank: u32,
        pod_name: &str,
        pod_ip: &str,
        nixl_md_bytes: Bytes,
    ) {
        let entry = ModelPeerEntry {
            agent_name: agent_name.to_string(),
            model_id: model_id.to_string(),
            tp_rank,
            pod_name: pod_name.to_string(),
            pod_ip: pod_ip.to_string(),
            registered_at: jiff::Timestamp::now().to_string(),
        };

        self.peer_registry
            .write()
            .await
            .insert(agent_name.to_string(), entry);

        self.nixl_metadata
            .write()
            .await
            .insert(agent_name.to_string(), nixl_md_bytes);

        self.persist_notify.notify_one();
    }

    pub async fn track_pod_agent(&self, pod_name: &str, agent_name: &str) {
        let mut pod_agents = self.pod_agents.write().await;
        pod_agents
            .entry(pod_name.to_string())
            .or_default()
            .push(agent_name.to_string());
    }

    pub async fn unregister_agent(&self, agent_name: &str) {
        self.peer_registry.write().await.remove(agent_name);
        self.nixl_metadata.write().await.remove(agent_name);

        let mut pod_agents = self.pod_agents.write().await;
        for agents in pod_agents.values_mut() {
            agents.retain(|a| a != agent_name);
        }
        pod_agents.retain(|_, v| !v.is_empty());

        self.persist_notify.notify_one();
    }

    pub async fn unregister_pod(&self, pod_name: &str) {
        let agents: Vec<String> = {
            let mut pod_agents = self.pod_agents.write().await;
            pod_agents.remove(pod_name).unwrap_or_default()
        };

        if agents.is_empty() {
            return;
        }

        info!(
            pod_name,
            agent_count = agents.len(),
            "cleaning up agents for deleted pod"
        );

        let mut registry = self.peer_registry.write().await;
        let mut metadata = self.nixl_metadata.write().await;
        for agent in &agents {
            registry.remove(agent);
            metadata.remove(agent);
        }

        self.persist_notify.notify_one();
    }

    pub async fn find_peers(&self, model_id: &str, tp_rank: u32) -> Vec<ModelPeerEntry> {
        self.peer_registry
            .read()
            .await
            .values()
            .filter(|e| e.model_id == model_id && e.tp_rank == tp_rank)
            .cloned()
            .collect()
    }

    pub async fn get_nixl_metadata(&self, agent_name: &str) -> Option<Bytes> {
        self.nixl_metadata.read().await.get(agent_name).cloned()
    }

    pub async fn snapshot_peers(&self) -> Vec<ModelPeerEntry> {
        self.peer_registry.read().await.values().cloned().collect()
    }

    pub async fn restore_from_snapshot(&self, peers: Vec<ModelPeerEntry>) {
        let mut registry = self.peer_registry.write().await;
        let mut pod_agents = self.pod_agents.write().await;

        registry.clear();
        pod_agents.clear();

        for peer in peers {
            pod_agents
                .entry(peer.pod_name.clone())
                .or_default()
                .push(peer.agent_name.clone());
            registry.insert(peer.agent_name.clone(), peer);
        }

        info!(peer_count = registry.len(), "restored state from ConfigMap");
    }
}

#[derive(Serialize, Deserialize)]
pub struct PersistedState {
    pub leader_identity: String,
    pub last_sync: String,
    pub peers: Vec<ModelPeerEntry>,
}

pub async fn persist_loop(
    state: Arc<ServerState>,
    client: kube::Client,
    namespace: String,
    configmap_name: String,
    leader_identity: String,
) {
    let cms: kube::Api<k8s_openapi::api::core::v1::ConfigMap> =
        kube::Api::namespaced(client, &namespace);

    loop {
        state.persist_notify.notified().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        let peers = state.snapshot_peers().await;
        let persisted = PersistedState {
            leader_identity: leader_identity.clone(),
            last_sync: jiff::Timestamp::now().to_string(),
            peers,
        };

        let data = match serde_json::to_string_pretty(&persisted) {
            Ok(d) => d,
            Err(e) => {
                error!(error = %e, "failed to serialize state for ConfigMap");
                continue;
            }
        };

        let cm = k8s_openapi::api::core::v1::ConfigMap {
            metadata: kube::api::ObjectMeta {
                name: Some(configmap_name.clone()),
                namespace: Some(namespace.clone()),
                ..Default::default()
            },
            data: Some({
                let mut m = std::collections::BTreeMap::new();
                m.insert("peers.json".to_string(), data);
                m
            }),
            ..Default::default()
        };

        match cms
            .patch(
                &configmap_name,
                &kube::api::PatchParams::apply("layercast-server"),
                &kube::api::Patch::Apply(cm),
            )
            .await
        {
            Ok(_) => {
                debug!(
                    peer_count = persisted.peers.len(),
                    "persisted state to ConfigMap"
                );
            }
            Err(e) => {
                warn!(error = %e, "failed to persist state to ConfigMap");
            }
        }
    }
}

pub async fn load_persisted_state(
    client: &kube::Client,
    namespace: &str,
    configmap_name: &str,
) -> Option<PersistedState> {
    let cms: kube::Api<k8s_openapi::api::core::v1::ConfigMap> =
        kube::Api::namespaced(client.clone(), namespace);

    match cms.get_opt(configmap_name).await {
        Ok(Some(cm)) => {
            let data = cm.data?.get("peers.json")?.clone();
            match serde_json::from_str::<PersistedState>(&data) {
                Ok(state) => {
                    info!(
                        peer_count = state.peers.len(),
                        leader = %state.leader_identity,
                        last_sync = %state.last_sync,
                        "loaded persisted state from ConfigMap"
                    );
                    Some(state)
                }
                Err(e) => {
                    warn!(error = %e, "failed to parse ConfigMap state, starting fresh");
                    None
                }
            }
        }
        Ok(None) => {
            info!("no persisted state ConfigMap found, starting fresh");
            None
        }
        Err(e) => {
            warn!(error = %e, "failed to read ConfigMap, starting fresh");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::state::{CachedModelInfo, FileListCache, ServerState};
    use bytes::Bytes;

    #[tokio::test]
    async fn register_and_find_peers() {
        let state = ServerState::new();

        state
            .register_peer(
                "agent-0",
                "Qwen/Qwen2.5-32B",
                0,
                "pod-0",
                "10.0.0.1",
                Bytes::from_static(b"nixl-data"),
            )
            .await;

        let peers = state.find_peers("Qwen/Qwen2.5-32B", 0).await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].agent_name, "agent-0");
        assert_eq!(peers[0].pod_ip, "10.0.0.1");

        // Different tp_rank should not match
        assert!(state.find_peers("Qwen/Qwen2.5-32B", 1).await.is_empty());
        // Different model should not match
        assert!(state.find_peers("other/model", 0).await.is_empty());
    }

    #[tokio::test]
    async fn unregister_agent_is_idempotent() {
        let state = ServerState::new();

        state
            .register_peer("agent-0", "m", 0, "pod-0", "10.0.0.1", Bytes::new())
            .await;
        state.track_pod_agent("pod-0", "agent-0").await;

        state.unregister_agent("agent-0").await;
        assert!(state.find_peers("m", 0).await.is_empty());
        assert!(state.get_nixl_metadata("agent-0").await.is_none());

        // Second call is a no-op
        state.unregister_agent("agent-0").await;
    }

    #[tokio::test]
    async fn unregister_pod_cleans_all_agents() {
        let state = ServerState::new();

        state
            .register_peer("rank0", "m", 0, "pod-0", "10.0.0.1", Bytes::from("a"))
            .await;
        state.track_pod_agent("pod-0", "rank0").await;
        state
            .register_peer("rank1", "m", 1, "pod-0", "10.0.0.1", Bytes::from("b"))
            .await;
        state.track_pod_agent("pod-0", "rank1").await;

        // Both agents registered
        assert_eq!(state.find_peers("m", 0).await.len(), 1);
        assert_eq!(state.find_peers("m", 1).await.len(), 1);

        state.unregister_pod("pod-0").await;

        assert!(state.find_peers("m", 0).await.is_empty());
        assert!(state.find_peers("m", 1).await.is_empty());
        assert!(state.get_nixl_metadata("rank0").await.is_none());
        assert!(state.get_nixl_metadata("rank1").await.is_none());
    }

    #[tokio::test]
    async fn unregister_pod_is_idempotent() {
        let state = ServerState::new();
        state.unregister_pod("ghost").await; // no-op
    }

    #[tokio::test]
    async fn nixl_metadata_roundtrip() {
        let state = ServerState::new();

        state
            .register_peer(
                "agent-0",
                "m",
                0,
                "pod-0",
                "10.0.0.1",
                Bytes::from_static(b"\xCA\xFE"),
            )
            .await;

        let md = state.get_nixl_metadata("agent-0").await.unwrap();
        assert_eq!(md.as_ref(), b"\xCA\xFE");
    }

    #[tokio::test]
    async fn snapshot_and_restore() {
        let state = ServerState::new();

        state
            .register_peer("a", "m1", 0, "pod-0", "10.0.0.1", Bytes::new())
            .await;
        state.track_pod_agent("pod-0", "a").await;
        state
            .register_peer("b", "m2", 1, "pod-1", "10.0.0.2", Bytes::new())
            .await;
        state.track_pod_agent("pod-1", "b").await;

        let snapshot = state.snapshot_peers().await;
        assert_eq!(snapshot.len(), 2);

        // Restore into a fresh state
        let fresh = ServerState::new();
        fresh.restore_from_snapshot(snapshot).await;

        assert_eq!(fresh.find_peers("m1", 0).await.len(), 1);
        assert_eq!(fresh.find_peers("m2", 1).await.len(), 1);

        // NIXL metadata is NOT restored (by design)
        assert!(fresh.get_nixl_metadata("a").await.is_none());
    }

    #[tokio::test]
    async fn file_cache_miss_and_hit() {
        let cache = FileListCache::new();
        assert!(cache.get("m", "main").await.is_none());

        cache
            .insert(
                "m",
                "main",
                CachedModelInfo::from_files(vec!["a.safetensors".into()]),
            )
            .await;
        let info = cache.get("m", "main").await.unwrap();
        assert_eq!(info.files, vec!["a.safetensors"]);
        assert!(info.weight_map.is_empty());

        // Different revision is a miss
        assert!(cache.get("m", "dev").await.is_none());
    }

    #[tokio::test]
    async fn register_overwrites_existing() {
        let state = ServerState::new();

        state
            .register_peer("agent-0", "m1", 0, "pod-0", "10.0.0.1", Bytes::from("old"))
            .await;
        state
            .register_peer("agent-0", "m2", 0, "pod-0", "10.0.0.1", Bytes::from("new"))
            .await;

        // Should have the new model
        let peers = state.find_peers("m2", 0).await;
        assert_eq!(peers.len(), 1);
        assert!(state.find_peers("m1", 0).await.is_empty());

        let md = state.get_nixl_metadata("agent-0").await.unwrap();
        assert_eq!(md.as_ref(), b"new");
    }
}
