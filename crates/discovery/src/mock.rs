use std::collections::HashMap;
use std::net::SocketAddr;

use tokio::sync::RwLock;

use crate::peer_discovery::{CompileCachePeer, ModelPeer};

/// Model peer advertisement stored in the mock.
#[derive(Debug, Clone)]
struct ModelPeerEntry {
    agent_name: String,
    model: String,
    tp_rank: u32,
}

#[derive(Debug, Default)]
struct MockState {
    /// agent_name -> ModelPeerEntry
    model_peers: HashMap<String, ModelPeerEntry>,
    compile_cache_namespace: Option<String>,
}

/// In-memory `PeerDiscovery` implementation for tests.
///
/// Not a mock in the "behavior fake" sense. This is a real in-memory store
/// that supports the full PeerDiscovery trait contract. Tests can write
/// advertisements and read them back without needing a k8s cluster.
pub struct MockDiscovery {
    node_id: String,
    peer_addr: SocketAddr,
    state: RwLock<MockState>,
}

impl MockDiscovery {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            peer_addr: "127.0.0.1:7901".parse().unwrap(),
            state: RwLock::new(MockState::default()),
        }
    }

    pub fn with_addr(node_id: impl Into<String>, addr: SocketAddr) -> Self {
        Self {
            node_id: node_id.into(),
            peer_addr: addr,
            state: RwLock::new(MockState::default()),
        }
    }
}

#[async_trait::async_trait]
impl crate::PeerDiscovery for MockDiscovery {
    async fn find_model_peers(&self, repo_id: &str, tp_rank: u32) -> Vec<ModelPeer> {
        let state = self.state.read().await;
        state
            .model_peers
            .values()
            .filter(|e| e.model == repo_id && e.tp_rank == tp_rank)
            .map(|e| ModelPeer {
                node_id: self.node_id.clone(),
                agent_name: e.agent_name.clone(),
                tp_rank: e.tp_rank,
                peer_addr: self.peer_addr,
            })
            .collect()
    }

    async fn advertise_model_peer(&self, agent_name: &str, model: &str, tp_rank: u32) {
        let mut state = self.state.write().await;
        state.model_peers.insert(
            agent_name.to_string(),
            ModelPeerEntry {
                agent_name: agent_name.to_string(),
                model: model.to_string(),
                tp_rank,
            },
        );
    }

    async fn unadvertise_model_peer(&self, agent_name: &str) {
        let mut state = self.state.write().await;
        state.model_peers.remove(agent_name);
    }

    async fn live_nodes(&self) -> Vec<String> {
        vec![self.node_id.clone()]
    }

    fn node_id(&self) -> &str {
        &self.node_id
    }

    async fn find_compile_cache_peers(&self, namespace: &str) -> Vec<CompileCachePeer> {
        let state = self.state.read().await;
        if state.compile_cache_namespace.as_deref() == Some(namespace) {
            vec![CompileCachePeer {
                node_id: self.node_id.clone(),
                peer_ip: self.peer_addr.ip(),
            }]
        } else {
            vec![]
        }
    }

    async fn advertise_compile_cache(&self, namespace: &str) {
        let mut state = self.state.write().await;
        state.compile_cache_namespace = Some(namespace.to_string());
    }

    async fn unadvertise_compile_cache(&self) {
        let mut state = self.state.write().await;
        state.compile_cache_namespace = None;
    }
}

#[cfg(test)]
mod tests {
    use crate::PeerDiscovery;
    use crate::mock::MockDiscovery;

    #[tokio::test]
    async fn model_peer_advertise_and_find() {
        let mock = MockDiscovery::new("test-node");
        mock.advertise_model_peer("agent-0", "meta-llama/70b", 0)
            .await;
        mock.advertise_model_peer("agent-1", "meta-llama/70b", 1)
            .await;

        let peers = mock.find_model_peers("meta-llama/70b", 0).await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].agent_name, "agent-0");

        let peers_r1 = mock.find_model_peers("meta-llama/70b", 1).await;
        assert_eq!(peers_r1.len(), 1);
        assert_eq!(peers_r1[0].agent_name, "agent-1");

        mock.unadvertise_model_peer("agent-0").await;
        assert!(mock.find_model_peers("meta-llama/70b", 0).await.is_empty());
    }

    #[tokio::test]
    async fn live_nodes_returns_self() {
        let mock = MockDiscovery::new("my-node");
        let nodes = mock.live_nodes().await;
        assert_eq!(nodes, vec!["my-node"]);
        assert_eq!(mock.node_id(), "my-node");
    }

    #[tokio::test]
    async fn compile_cache_advertise_and_find() {
        let mock = MockDiscovery::new("gpu-node-1");
        let ns = "H100-SXM5:sha256:abc:meta-llama/70b";

        // No peers before advertising
        assert!(mock.find_compile_cache_peers(ns).await.is_empty());

        // Advertise and find
        mock.advertise_compile_cache(ns).await;
        let peers = mock.find_compile_cache_peers(ns).await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].node_id, "gpu-node-1");
        assert_eq!(
            peers[0].peer_ip,
            "127.0.0.1".parse::<std::net::IpAddr>().unwrap()
        );

        // Wrong namespace returns nothing
        assert!(
            mock.find_compile_cache_peers("A100:sha256:xyz:other/model")
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn compile_cache_unadvertise() {
        let mock = MockDiscovery::new("gpu-node-2");
        let ns = "H100-SXM5:sha256:abc:meta-llama/70b";

        mock.advertise_compile_cache(ns).await;
        assert_eq!(mock.find_compile_cache_peers(ns).await.len(), 1);

        mock.unadvertise_compile_cache().await;
        assert!(mock.find_compile_cache_peers(ns).await.is_empty());
    }

    #[test]
    fn nodecache_status_backward_compat() {
        use crate::crd::NodeCacheStatus;

        // JSON without compile_cache_namespace should deserialize fine (defaults to None)
        let json = r#"{"model_peers": [{"agent_name": "a", "model": "m", "tp_rank": 0}]}"#;
        let status: NodeCacheStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.model_peers.len(), 1);
        assert!(status.compile_cache_namespace.is_none());

        // Empty JSON
        let status: NodeCacheStatus = serde_json::from_str("{}").unwrap();
        assert!(status.compile_cache_namespace.is_none());

        // With compile_cache_namespace present
        let json = r#"{"model_peers": [], "compile_cache_namespace": "H100:sha256:abc"}"#;
        let status: NodeCacheStatus = serde_json::from_str(json).unwrap();
        assert_eq!(
            status.compile_cache_namespace.as_deref(),
            Some("H100:sha256:abc")
        );
    }
}
