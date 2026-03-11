//! In-memory store for NIXL VRAM metadata blobs.
//!
//! NIXL agent metadata is too large for the PodCache CRD (often >100KB).
//! Instead, the daemon stores metadata here and serves it via HTTP. The CRD
//! carries a lightweight pointer (model, tp_rank, agent_name).
//!
//!   publish flow:  vLLM plugin --IPC--> daemon stores in ModelPeerStore
//!                                       + advertises pointer via PodCache CRD
//!
//!   fetch flow:    peer discovers pointer via PodCache reflector
//!                  --> HTTP GET /internal/nixl-vram/{agent_name}
//!                  --> returns raw metadata bytes

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::RwLock;

/// Thread-safe store mapping agent_name -> raw NIXL metadata bytes.
///
/// Uses `Bytes` internally so clones are cheap refcount bumps (no memcpy).
/// The HTTP handler can serve directly from the same allocation.
#[derive(Clone, Default)]
pub struct ModelPeerStore {
    inner: Arc<RwLock<HashMap<String, Bytes>>>,
}

impl ModelPeerStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store metadata for a NIXL agent. Overwrites any existing entry.
    pub async fn insert(&self, agent_name: &str, metadata: Vec<u8>) {
        self.inner
            .write()
            .await
            .insert(agent_name.to_string(), Bytes::from(metadata));
    }

    /// Retrieve metadata for a NIXL agent. Returns None if not found.
    /// Cloning `Bytes` is a cheap refcount bump.
    pub async fn get(&self, agent_name: &str) -> Option<Bytes> {
        self.inner.read().await.get(agent_name).cloned()
    }

    /// Remove metadata for a NIXL agent.
    pub async fn remove(&self, agent_name: &str) {
        self.inner.write().await.remove(agent_name);
    }
}

#[cfg(test)]
mod tests {
    use crate::nixl_vram_store::ModelPeerStore;

    #[tokio::test]
    async fn insert_get_remove() {
        let store = ModelPeerStore::new();

        assert!(store.get("agent-0").await.is_none());

        store.insert("agent-0", vec![1, 2, 3]).await;
        assert_eq!(store.get("agent-0").await.unwrap().as_ref(), &[1, 2, 3]);

        store.insert("agent-0", vec![4, 5]).await;
        assert_eq!(store.get("agent-0").await.unwrap().as_ref(), &[4, 5]);

        store.remove("agent-0").await;
        assert!(store.get("agent-0").await.is_none());
    }
}
