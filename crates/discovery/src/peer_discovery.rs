use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};

/// Information about a peer with NIXL VRAM metadata for a model.
///
/// Does NOT include the actual metadata bytes (those are too large for the CRD).
/// The caller must fetch metadata out-of-band from the peer's HTTP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPeer {
    pub node_id: String,
    pub agent_name: String,
    pub tp_rank: u32,
    /// The peer's advertise address (IP:port). Used to derive the
    /// HTTP endpoint for fetching the full NIXL metadata.
    pub peer_addr: SocketAddr,
}

/// A peer that can serve compile cache artifacts for a given namespace.
#[derive(Debug, Clone)]
pub struct CompileCachePeer {
    pub node_id: String,
    pub peer_ip: IpAddr,
}

/// Backend-agnostic peer discovery and advertisement.
///
/// Implementations:
/// - `K8sDiscovery`: production backend using PodCache CRD + kube-rs reflectors
/// - `MockDiscovery`: in-memory backend for tests
///
/// All consumers hold `Arc<dyn PeerDiscovery>` and never know which
/// backend is active.
#[async_trait::async_trait]
pub trait PeerDiscovery: Send + Sync {
    /// Find peers with NIXL VRAM metadata for a model + TP rank.
    async fn find_model_peers(&self, repo_id: &str, tp_rank: u32) -> Vec<ModelPeer>;

    /// Advertise model peer availability (lightweight pointer only).
    ///
    /// The actual metadata is stored out-of-band and served via HTTP.
    async fn advertise_model_peer(&self, agent_name: &str, model: &str, tp_rank: u32);

    /// Remove model peer advertisement.
    async fn unadvertise_model_peer(&self, agent_name: &str);

    /// List all live node IDs in the cluster (including self).
    async fn live_nodes(&self) -> Vec<String>;

    /// Node ID of this instance.
    fn node_id(&self) -> &str;

    /// Find peers that share the same compile cache namespace.
    async fn find_compile_cache_peers(&self, namespace: &str) -> Vec<CompileCachePeer>;

    /// Advertise that this node can serve compile cache for the given namespace.
    async fn advertise_compile_cache(&self, namespace: &str);

    /// Remove compile cache advertisement.
    async fn unadvertise_compile_cache(&self);
}
