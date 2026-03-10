use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use kube::api::{Api, ObjectMeta, Patch, PatchParams};
use kube::runtime::{WatchStreamExt, reflector, watcher};
use kube::{Client, ResourceExt};
use tokio::sync::{Mutex, Notify};
use tracing::{debug, info, warn};

use crate::crd::{ModelPeerEntry, NodeCache, NodeCacheSpec, NodeCacheStatus};
use crate::peer_discovery::{CompileCachePeer, ModelPeer};

/// Configuration for K8s-native peer discovery.
///
/// All fields typically come from the downward API (POD_NAME, POD_NAMESPACE,
/// NODE_NAME, POD_IP) and CLI/env config (ports).
#[derive(Debug, Clone)]
pub struct K8sDiscoveryConfig {
    pub pod_name: String,
    pub pod_namespace: String,
    pub node_name: String,
    pub pod_ip: String,
    pub nixl_control_port: u16,
}

/// Kubernetes-native peer discovery using NodeCache CRDs and kube-rs reflectors.
///
/// ```text
///  ┌────────────────────────────────┐
///  │        API Server              │
///  │  NodeCache objects (namespace) │
///  └──────────┬─────────────────────┘
///             │ watch
///             ▼
///  ┌────────────────────────────────┐
///  │ K8sDiscovery                   │
///  │  reflector store (in-memory)   │◄── queries read from store
///  │  own_status (Mutex)            │──► patch_status writes to API
///  └────────────────────────────────┘
/// ```
///
/// On startup: creates own NodeCache CR with ownerReference to the daemon Pod
/// (k8s GC deletes it when the pod dies). Starts a reflector to watch all
/// NodeCache objects in the namespace.
///
/// Queries (find_model_peers) read from the in-memory reflector store,
/// never hitting the API server.
///
/// Advertisements (advertise_model_peer) update a local shadow status and
/// patch the status subresource.
/// How long to wait after a status mutation before flushing to the API server.
/// Coalesces rapid-fire advertise calls into a single PATCH.
const STATUS_DEBOUNCE: Duration = Duration::from_millis(100);

pub struct K8sDiscovery {
    config: K8sDiscoveryConfig,
    node_cache_store: reflector::Store<NodeCache>,
    own_status: Arc<Mutex<NodeCacheStatus>>,
    /// Wakes the background status patcher when own_status is dirty.
    status_dirty: Arc<Notify>,
    _handles: Vec<tokio::task::JoinHandle<()>>,
}

impl K8sDiscovery {
    /// Start the K8s discovery backend.
    ///
    /// Creates the NodeCache CR, starts the reflector, and returns a ready
    /// K8sDiscovery that implements PeerDiscovery.
    pub async fn start(config: K8sDiscoveryConfig) -> Result<Self> {
        let client = Client::try_default()
            .await
            .context("failed to create kube client (are we running in-cluster?)")?;

        // Look up own pod for ownerReference UID
        let pods: Api<Pod> = Api::namespaced(client.clone(), &config.pod_namespace);
        let own_pod = pods
            .get(&config.pod_name)
            .await
            .context("failed to get own pod")?;
        let pod_uid = own_pod
            .metadata
            .uid
            .as_ref()
            .context("own pod has no UID")?
            .clone();

        // Create or update our NodeCache CR via server-side apply.
        // Keyed by pod name (not node name) so multiple daemons on the same
        // node each get their own CR instead of clobbering each other.
        let cr_name = format!("pod-{}", config.pod_name);
        let nc_api: Api<NodeCache> = Api::namespaced(client.clone(), &config.pod_namespace);

        let nc = NodeCache {
            metadata: ObjectMeta {
                name: Some(cr_name.clone()),
                namespace: Some(config.pod_namespace.clone()),
                owner_references: Some(vec![OwnerReference {
                    api_version: "v1".to_string(),
                    kind: "Pod".to_string(),
                    name: config.pod_name.clone(),
                    uid: pod_uid,
                    controller: Some(false),
                    block_owner_deletion: Some(false),
                }]),
                ..Default::default()
            },
            spec: NodeCacheSpec {
                node_name: config.node_name.clone(),
                pod_ip: config.pod_ip.clone(),
                nixl_control_port: config.nixl_control_port,
            },
            status: None,
        };

        nc_api
            .patch(
                &cr_name,
                &PatchParams::apply("layercast-daemon").force(),
                &Patch::Apply(nc),
            )
            .await
            .context("failed to create/update own NodeCache")?;

        info!(
            cr_name = %cr_name,
            node_name = %config.node_name,
            pod_ip = %config.pod_ip,
            "created own NodeCache CR"
        );

        // Start NodeCache reflector
        let (nc_store, nc_writer) = reflector::store();
        let nc_watcher = watcher(nc_api.clone(), watcher::Config::default());
        let nc_stream = reflector(nc_writer, nc_watcher).applied_objects();

        let nc_handle = tokio::spawn(async move {
            let mut stream = std::pin::pin!(nc_stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(nc) => {
                        debug!(
                            name = nc.name_any(),
                            node = %nc.spec.node_name,
                            "NodeCache updated in reflector store"
                        );
                    }
                    Err(e) => {
                        warn!(error = %e, "NodeCache watcher error (will retry)");
                    }
                }
            }
        });

        let own_status = Arc::new(Mutex::new(NodeCacheStatus::default()));
        let status_dirty = Arc::new(Notify::new());

        // Background task: waits for status_dirty notifications, debounces,
        // then flushes own_status to the API server in a single PATCH.
        let flush_client = client.clone();
        let flush_ns = config.pod_namespace.clone();
        let flush_cr = cr_name.clone();
        let flush_status = Arc::clone(&own_status);
        let flush_dirty = Arc::clone(&status_dirty);
        let flush_handle = tokio::spawn(async move {
            loop {
                flush_dirty.notified().await;
                // Debounce: wait a short window for more mutations to land
                tokio::time::sleep(STATUS_DEBOUNCE).await;

                let status_snapshot = flush_status.lock().await.clone();
                let nc_api: Api<NodeCache> = Api::namespaced(flush_client.clone(), &flush_ns);
                let patch = serde_json::json!({ "status": status_snapshot });
                if let Err(e) = nc_api
                    .patch_status(&flush_cr, &PatchParams::default(), &Patch::Merge(patch))
                    .await
                {
                    warn!(error = %e, "debounced status patch failed");
                } else {
                    debug!("flushed debounced status patch to API server");
                }
            }
        });

        Ok(Self {
            config,
            node_cache_store: nc_store,
            own_status,
            status_dirty,
            _handles: vec![nc_handle, flush_handle],
        })
    }
}

#[async_trait::async_trait]
impl crate::PeerDiscovery for K8sDiscovery {
    async fn find_model_peers(&self, repo_id: &str, tp_rank: u32) -> Vec<ModelPeer> {
        let nodes: Vec<Arc<NodeCache>> = self.node_cache_store.state();
        let mut peers = Vec::new();

        debug!(
            repo_id,
            tp_rank,
            cr_count = nodes.len(),
            "find_model_peers: scanning reflector store"
        );

        for nc in &nodes {
            let cr_name = nc.name_any();
            let status = match &nc.status {
                Some(s) => s,
                None => {
                    debug!(cr_name, "find_model_peers: CR has no status, skipping");
                    continue;
                }
            };

            debug!(
                cr_name,
                model_peer_count = status.model_peers.len(),
                "find_model_peers: checking CR"
            );

            for entry in &status.model_peers {
                if entry.model != repo_id {
                    continue;
                }
                if entry.tp_rank != tp_rank {
                    continue;
                }

                let peer_addr: SocketAddr =
                    format!("{}:{}", nc.spec.pod_ip, nc.spec.nixl_control_port)
                        .parse()
                        .unwrap_or_else(|_| ([0, 0, 0, 0], 0).into());
                peers.push(ModelPeer {
                    node_id: nc.spec.node_name.clone(),
                    agent_name: entry.agent_name.clone(),
                    tp_rank: entry.tp_rank,
                    peer_addr,
                });
            }
        }

        debug!(
            repo_id,
            tp_rank,
            found = peers.len(),
            "find_model_peers: result"
        );
        peers
    }

    async fn advertise_model_peer(&self, agent_name: &str, model: &str, tp_rank: u32) {
        {
            let mut status = self.own_status.lock().await;
            status.model_peers.retain(|e| e.agent_name != agent_name);
            status.model_peers.push(ModelPeerEntry {
                agent_name: agent_name.to_string(),
                model: model.to_string(),
                tp_rank,
            });
        }
        self.status_dirty.notify_one();
        debug!(
            agent_name,
            model, tp_rank, "queued model peer advertisement"
        );
    }

    async fn unadvertise_model_peer(&self, agent_name: &str) {
        {
            let mut status = self.own_status.lock().await;
            status.model_peers.retain(|e| e.agent_name != agent_name);
        }
        self.status_dirty.notify_one();
        debug!(agent_name, "queued model peer unadvertisement");
    }

    async fn live_nodes(&self) -> Vec<String> {
        self.node_cache_store
            .state()
            .iter()
            .map(|nc| nc.spec.node_name.clone())
            .collect()
    }

    fn node_id(&self) -> &str {
        &self.config.node_name
    }

    async fn find_compile_cache_peers(&self, namespace: &str) -> Vec<CompileCachePeer> {
        let nodes: Vec<Arc<NodeCache>> = self.node_cache_store.state();
        let mut peers = Vec::new();

        for nc in &nodes {
            let status = match &nc.status {
                Some(s) => s,
                None => continue,
            };

            if status.compile_cache_namespace.as_deref() == Some(namespace) {
                let ip = match nc.spec.pod_ip.parse() {
                    Ok(ip) => ip,
                    Err(_) => continue,
                };
                peers.push(CompileCachePeer {
                    node_id: nc.spec.node_name.clone(),
                    peer_ip: ip,
                });
            }
        }

        peers
    }

    async fn advertise_compile_cache(&self, namespace: &str) {
        {
            let mut status = self.own_status.lock().await;
            status.compile_cache_namespace = Some(namespace.to_string());
        }
        self.status_dirty.notify_one();
        debug!(namespace, "queued compile cache advertisement");
    }

    async fn unadvertise_compile_cache(&self) {
        {
            let mut status = self.own_status.lock().await;
            status.compile_cache_namespace = None;
        }
        self.status_dirty.notify_one();
        debug!("queued compile cache unadvertisement");
    }
}

#[cfg(test)]
mod tests {
    use crate::crd::*;

    #[test]
    fn model_peer_filtering_logic() {
        let entries = [
            ModelPeerEntry {
                agent_name: "agent-0".to_string(),
                model: "meta-llama/70b".to_string(),
                tp_rank: 0,
            },
            ModelPeerEntry {
                agent_name: "agent-1".to_string(),
                model: "meta-llama/70b".to_string(),
                tp_rank: 1,
            },
            ModelPeerEntry {
                agent_name: "agent-2".to_string(),
                model: "other/model".to_string(),
                tp_rank: 0,
            },
        ];

        // Filter like find_model_peers does
        let repo_id = "meta-llama/70b";
        let tp_rank = 0u32;

        let matched: Vec<&ModelPeerEntry> = entries
            .iter()
            .filter(|e| e.model == repo_id && e.tp_rank == tp_rank)
            .collect();

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].agent_name, "agent-0");

        // tp_rank=1
        let matched: Vec<&ModelPeerEntry> = entries
            .iter()
            .filter(|e| e.model == repo_id && e.tp_rank == 1)
            .collect();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].agent_name, "agent-1");

        // wrong model
        let matched: Vec<&ModelPeerEntry> = entries
            .iter()
            .filter(|e| e.model == "nonexistent" && e.tp_rank == 0)
            .collect();
        assert!(matched.is_empty());
    }
}
