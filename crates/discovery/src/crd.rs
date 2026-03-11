use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// PodCache represents the state of a layercast sidecar on a single pod.
///
/// One CR per pod, keyed as `pod-{pod_name}`. Lifecycle is tied to the pod
/// via ownerReference (K8s GC deletes the CR when the pod dies).
///
/// Spec is written once at boot (identity, ports). Status is updated as
/// model peers are registered, using the status subresource for cheap
/// partial writes.
///
/// ```text
///  ┌──────────────────────────────────────┐
///  │  PodCache: pod-vllm-seed-0          │
///  │  ownerRef → vllm-seed-0             │
///  ├──────────────────────────────────────┤
///  │ spec (stable after boot):            │
///  │   nodeName, podIP, nixlControlPort   │
///  ├──────────────────────────────────────┤
///  │ status (updated on peer events):     │
///  │   modelPeers: [{agent, model, ...}]  │
///  └──────────────────────────────────────┘
/// ```
#[derive(CustomResource, Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[kube(
    group = "layercast.io",
    version = "v1alpha1",
    kind = "PodCache",
    namespaced,
    status = "PodCacheStatus",
    printcolumn = r#"{"name":"Node","type":"string","jsonPath":".spec.nodeName"}"#,
    printcolumn = r#"{"name":"IP","type":"string","jsonPath":".spec.podIP"}"#
)]
pub struct PodCacheSpec {
    pub node_name: String,
    pub pod_ip: String,
    pub nixl_control_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct PodCacheStatus {
    /// Model peer metadata relayed from local vLLM pods.
    #[serde(default)]
    pub model_peers: Vec<ModelPeerEntry>,

    /// Compile cache namespace this node can serve (e.g. "H100-SXM5:sha256:abc:meta-llama/70b").
    /// Peers with the same namespace can share torch.compile artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_cache_namespace: Option<String>,
}

/// Lightweight pointer stored in PodCache CRD status.
///
/// The file list is intentionally omitted: it's deterministic given
/// (model, tp_rank) and already cached locally via FileListCache.
/// Keeping the CRD lean lets us add new fields (like compile cache
/// namespace) without bloating the object.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModelPeerEntry {
    pub agent_name: String,
    pub model: String,
    pub tp_rank: u32,
}

#[cfg(test)]
mod tests {
    use crate::crd::*;

    #[test]
    fn podcache_spec_roundtrip() {
        let spec = PodCacheSpec {
            node_name: "worker-3".to_string(),
            pod_ip: "10.0.0.42".to_string(),
            nixl_control_port: 7903,
        };

        let json = serde_json::to_string(&spec).unwrap();
        let decoded: PodCacheSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.node_name, "worker-3");
        assert_eq!(decoded.nixl_control_port, 7903);
    }

    #[test]
    fn podcache_status_roundtrip() {
        let status = PodCacheStatus {
            model_peers: vec![ModelPeerEntry {
                agent_name: "layercast-abc123".to_string(),
                model: "Qwen/Qwen2.5-3B".to_string(),
                tp_rank: 0,
            }],
            compile_cache_namespace: None,
        };

        let json = serde_json::to_string(&status).unwrap();
        let decoded: PodCacheStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.model_peers.len(), 1);
        assert_eq!(decoded.model_peers[0].tp_rank, 0);
    }

    #[test]
    fn podcache_status_defaults_empty() {
        let status: PodCacheStatus = serde_json::from_str("{}").unwrap();
        assert!(status.model_peers.is_empty());
    }

    #[test]
    fn crd_generation() {
        use kube::CustomResourceExt;
        let crd = PodCache::crd();
        assert_eq!(crd.metadata.name.as_deref(), Some("podcaches.layercast.io"));
    }
}
