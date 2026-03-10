pub mod crd;
pub mod inference_pool;
pub mod k8s;
pub mod mock;
pub mod peer_discovery;

pub use crate::mock::MockDiscovery;
pub use crate::peer_discovery::{CompileCachePeer, ModelPeer, PeerDiscovery};

pub use crate::k8s::{K8sDiscovery, K8sDiscoveryConfig};
