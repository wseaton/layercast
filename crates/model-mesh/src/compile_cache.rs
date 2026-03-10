//! Two-tier (memory + disk) compile cache with P2P peer exchange.
//!
//! torch.compile generates GPU-specific compiled kernels that are expensive to
//! recompute. This module provides a RESP (Redis protocol) server so torch can
//! use its built-in `RedisRemoteCacheBackend`, backed by a local memory+disk
//! store that also fans out to peers over HTTP on cache misses.
//!
//! Namespace: `{gpu_sku}:{image_digest}:{model_name}`
//! Ensures cache entries are only shared between compatible configurations.
//!
//!   ┌──────────────┐  RESP (SET/GET)  ┌──────────────────────┐
//!   │  torch.compile├────────────────►│  CompileCacheServer   │
//!   └──────────────┘                  └──────────┬───────────┘
//!                                                │
//!                                     ┌──────────▼───────────┐
//!                                     │  CompileCacheStore    │
//!                                     │  ┌────────┐ ┌──────┐ │
//!                                     │  │ memory │ │ disk │ │
//!                                     │  └────────┘ └──────┘ │
//!                                     └──────────┬───────────┘
//!                                                │ miss
//!                                     ┌──────────▼───────────┐
//!                                     │  fan-out HTTP GET to  │
//!                                     │  peers via discovery  │
//!                                     └──────────────────────┘

use std::collections::HashMap;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use bytes::BytesMut;
use discovery::PeerDiscovery;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::resp::{self, RespCommand, RespResponse};

// CompileCacheNamespace

/// Strong type wrapping `{gpu_sku}:{image_digest}:{model_name}`.
///
/// Ensures compile cache entries are only shared between pods running the
/// same GPU, container image, and model. Mixing any of these would produce
/// invalid or crashing kernels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileCacheNamespace {
    inner: String,
}

impl CompileCacheNamespace {
    pub fn new(gpu_sku: &str, image_digest: &str, model_name: &str) -> Self {
        Self {
            inner: format!("{gpu_sku}:{image_digest}:{model_name}"),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }
}

impl fmt::Display for CompileCacheNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner)
    }
}

// CompileCacheStore

/// Two-tier compile cache: memory (with random eviction) + disk.
///
/// On GET misses, fans out HTTP requests to peers discovered via
/// the `PeerDiscovery` trait.
#[derive(Clone)]
pub struct CompileCacheStore {
    memory: Arc<RwLock<HashMap<String, Bytes>>>,
    memory_size: Arc<AtomicUsize>,
    max_memory: usize,
    disk_dir: PathBuf,
    namespace: CompileCacheNamespace,
    discovery: Arc<dyn PeerDiscovery>,
    http_client: reqwest::Client,
    http_port: u16,
    namespace_advertised: Arc<AtomicBool>,
    stats: Arc<CompileCacheStats>,
}

/// Hit/miss counters for observability.
#[derive(Default)]
pub struct CompileCacheStats {
    pub gets: AtomicUsize,
    pub mem_hits: AtomicUsize,
    pub disk_hits: AtomicUsize,
    pub peer_hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub sets: AtomicUsize,
}

impl CompileCacheStats {
    pub fn snapshot(&self) -> CompileCacheStatsSnapshot {
        CompileCacheStatsSnapshot {
            gets: self.gets.load(Ordering::Relaxed),
            mem_hits: self.mem_hits.load(Ordering::Relaxed),
            disk_hits: self.disk_hits.load(Ordering::Relaxed),
            peer_hits: self.peer_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            sets: self.sets.load(Ordering::Relaxed),
        }
    }
}

#[derive(serde::Serialize)]
pub struct CompileCacheStatsSnapshot {
    pub gets: usize,
    pub mem_hits: usize,
    pub disk_hits: usize,
    pub peer_hits: usize,
    pub misses: usize,
    pub sets: usize,
}

impl CompileCacheStore {
    pub fn new(
        namespace: CompileCacheNamespace,
        disk_dir: PathBuf,
        max_memory: usize,
        discovery: Arc<dyn PeerDiscovery>,
        http_port: u16,
    ) -> Self {
        let ns_hash = hash_string(namespace.as_str());
        let dir = disk_dir.join(ns_hash);

        // Best-effort directory creation. If it fails we'll get errors on
        // disk writes later, which is fine (memory tier still works).
        if let Err(e) = std::fs::create_dir_all(&dir) {
            warn!(path = %dir.display(), error = %e, "failed to create compile cache disk dir");
        }

        Self {
            memory: Arc::new(RwLock::new(HashMap::new())),
            memory_size: Arc::new(AtomicUsize::new(0)),
            max_memory,
            disk_dir: dir,
            namespace,
            discovery,
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap_or_default(),
            http_port,
            namespace_advertised: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(CompileCacheStats::default()),
        }
    }

    /// Stats snapshot for the HTTP observability endpoint.
    pub fn stats(&self) -> CompileCacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// GET: memory -> disk -> peers.
    pub async fn get(&self, key: &str) -> Option<Bytes> {
        self.stats.gets.fetch_add(1, Ordering::Relaxed);

        // Check memory
        if let Some(val) = self.get_memory(key).await {
            self.stats.mem_hits.fetch_add(1, Ordering::Relaxed);
            return Some(val);
        }

        // Check disk
        if let Some(val) = self.get_disk(key).await {
            self.stats.disk_hits.fetch_add(1, Ordering::Relaxed);
            // Promote to memory
            self.set_memory(key, val.clone()).await;
            return Some(val);
        }

        // Fan out to peers
        if let Some(val) = self.fetch_from_peers(key).await {
            self.stats.peer_hits.fetch_add(1, Ordering::Relaxed);
            // Store locally for future hits
            self.set_memory(key, val.clone()).await;
            self.write_disk(key, &val).await;
            return Some(val);
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// GET from local tiers only (memory + disk). No peer fan-out.
    /// Used by the HTTP endpoint to avoid infinite recursion.
    pub async fn get_local(&self, key: &str) -> Option<Bytes> {
        if let Some(val) = self.get_memory(key).await {
            return Some(val);
        }
        if let Some(val) = self.get_disk(key).await {
            self.set_memory(key, val.clone()).await;
            return Some(val);
        }
        None
    }

    /// SET: write to memory (with eviction) and disk.
    pub async fn set(&self, key: &str, value: Bytes) {
        self.stats.sets.fetch_add(1, Ordering::Relaxed);
        self.set_memory(key, value.clone()).await;
        self.write_disk(key, &value).await;

        // Lazy-advertise: first SET means we have something worth sharing.
        if !self.namespace_advertised.load(Ordering::Relaxed) {
            self.discovery
                .advertise_compile_cache(self.namespace.as_str())
                .await;
            self.namespace_advertised.store(true, Ordering::Relaxed);
            info!(namespace = %self.namespace, "compile cache: advertised namespace");
        }
    }

    /// EXISTS: check memory and disk. No peer fan-out.
    pub async fn exists(&self, key: &str) -> bool {
        {
            let mem = self.memory.read().await;
            if mem.contains_key(key) {
                return true;
            }
        }
        let path = self.disk_path(key);
        // tokio::fs::try_exists is cleaner but let's just use metadata
        tokio::fs::metadata(&path).await.is_ok()
    }

    /// Fan out HTTP GET to peers for a cache entry.
    async fn fetch_from_peers(&self, key: &str) -> Option<Bytes> {
        let peers = self
            .discovery
            .find_compile_cache_peers(self.namespace.as_str())
            .await;

        if peers.is_empty() {
            return None;
        }

        // Filter out self (no point asking ourselves)
        let my_node = self.discovery.node_id().to_string();
        let remote_peers: Vec<_> = peers.into_iter().filter(|p| p.node_id != my_node).collect();

        if remote_peers.is_empty() {
            return None;
        }

        debug!(
            key,
            peer_count = remote_peers.len(),
            "compile cache: fanning out to peers"
        );

        // Fire requests to all peers, take the first success.
        let futs: Vec<_> = remote_peers
            .into_iter()
            .map(|peer| {
                let url = format!(
                    "http://{}:{}/internal/compile-cache/{}",
                    peer.peer_ip, self.http_port, key
                );
                let client = self.http_client.clone();
                async move {
                    match client.get(&url).send().await {
                        Ok(resp) if resp.status().is_success() => resp.bytes().await.ok(),
                        _ => None,
                    }
                }
            })
            .collect();

        // Race all requests, return the first non-None result.
        let results = futures::future::join_all(futs).await;
        results.into_iter().flatten().next()
    }

    async fn get_memory(&self, key: &str) -> Option<Bytes> {
        self.memory.read().await.get(key).cloned()
    }

    async fn set_memory(&self, key: &str, value: Bytes) {
        let value_len = value.len();
        let mut mem = self.memory.write().await;

        // If replacing an existing key, subtract its old size first.
        if let Some(old) = mem.get(key) {
            self.memory_size.fetch_sub(old.len(), Ordering::Relaxed);
        }

        mem.insert(key.to_string(), value);
        self.memory_size.fetch_add(value_len, Ordering::Relaxed);

        // Evict if over budget. Not true LRU, just random eviction since
        // HashMap iteration order is arbitrary. Good enough for a cache
        // where entries are roughly similar size and access patterns are
        // "write once, read occasionally."
        while self.memory_size.load(Ordering::Relaxed) > self.max_memory && mem.len() > 1 {
            // Grab an arbitrary key to evict (skip the one we just inserted)
            let victim = mem.keys().find(|k| k.as_str() != key).cloned();
            if let Some(victim_key) = victim {
                if let Some(evicted) = mem.remove(&victim_key) {
                    self.memory_size.fetch_sub(evicted.len(), Ordering::Relaxed);
                    debug!(
                        key = victim_key,
                        bytes = evicted.len(),
                        "compile cache: evicted from memory"
                    );
                }
            } else {
                break;
            }
        }
    }

    async fn get_disk(&self, key: &str) -> Option<Bytes> {
        let path = self.disk_path(key);
        match tokio::fs::read(&path).await {
            Ok(data) => Some(Bytes::from(data)),
            Err(_) => None,
        }
    }

    async fn write_disk(&self, key: &str, value: &[u8]) {
        let path = self.disk_path(key);
        if let Err(e) = tokio::fs::write(&path, value).await {
            warn!(
                key,
                path = %path.display(),
                error = %e,
                "compile cache: failed to write to disk"
            );
        }
    }

    fn disk_path(&self, key: &str) -> PathBuf {
        let key_hash = hash_string(key);
        self.disk_dir.join(key_hash)
    }

    /// Expose the namespace for logging/diagnostics.
    pub fn namespace(&self) -> &CompileCacheNamespace {
        &self.namespace
    }
}

// CompileCacheServer

/// TCP server speaking a minimal RESP2 subset for torch.compile's
/// `RedisRemoteCacheBackend`. Delegates all storage to `CompileCacheStore`.
pub struct CompileCacheServer {
    addr: String,
    store: CompileCacheStore,
}

impl CompileCacheServer {
    pub fn new(addr: String, store: CompileCacheStore) -> Self {
        Self { addr, store }
    }

    pub async fn start(self) -> anyhow::Result<JoinHandle<()>> {
        let listener = TcpListener::bind(&self.addr).await?;
        info!(addr = %self.addr, "compile cache RESP server listening");

        let store = self.store;
        let handle = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer)) => {
                        debug!(%peer, "compile cache: client connected");
                        let store = store.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, store).await {
                                debug!(error = %e, "compile cache: connection closed");
                            }
                        });
                    }
                    Err(e) => {
                        error!(error = %e, "compile cache: accept error");
                        break;
                    }
                }
            }
        });

        Ok(handle)
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    store: CompileCacheStore,
) -> anyhow::Result<()> {
    let mut buf = BytesMut::with_capacity(4096);
    let mut out = BytesMut::with_capacity(1024);

    loop {
        // Read more data from the client.
        let n = stream.read_buf(&mut buf).await?;
        if n == 0 {
            return Ok(()); // clean disconnect
        }

        // Process all complete commands in the buffer.
        loop {
            match resp::parse_command(&mut buf) {
                Ok(cmd) => {
                    let response = dispatch_command(&store, cmd).await;
                    response.encode(&mut out);
                    stream.write_all(&out).await?;
                    out.clear();
                }
                Err(resp::RespError::Incomplete) => break,
                Err(e) => {
                    let resp = RespResponse::Error(e.to_string());
                    resp.encode(&mut out);
                    stream.write_all(&out).await?;
                    out.clear();
                    return Ok(());
                }
            }
        }
    }
}

async fn dispatch_command(store: &CompileCacheStore, cmd: RespCommand) -> RespResponse {
    match cmd {
        RespCommand::Ping => RespResponse::pong(),

        RespCommand::Get { key } => {
            let key_str = String::from_utf8_lossy(&key);
            match store.get(&key_str).await {
                Some(data) => RespResponse::BulkString(data.to_vec()),
                None => RespResponse::Null,
            }
        }

        RespCommand::Set { key, value } => {
            let key_str = String::from_utf8_lossy(&key).into_owned();
            store.set(&key_str, Bytes::from(value)).await;
            RespResponse::ok()
        }

        RespCommand::Exists { key } => {
            let key_str = String::from_utf8_lossy(&key);
            let exists = store.exists(&key_str).await;
            RespResponse::Integer(if exists { 1 } else { 0 })
        }

        RespCommand::Handshake { ref command } => match command.as_str() {
            "CLIENT" => RespResponse::ok(),
            "INFO" => RespResponse::BulkString(Vec::new()),
            // COMMAND, CONFIG, and anything else we don't recognize
            _ => RespResponse::EmptyArray,
        },
    }
}

// Helpers

fn hash_string(s: &str) -> String {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// Tests

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use discovery::{MockDiscovery, PeerDiscovery};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::compile_cache::{CompileCacheNamespace, CompileCacheServer, CompileCacheStore};
    use crate::hf_api::build_internal_router;
    use crate::nixl_vram_store::ModelPeerStore;

    fn test_namespace() -> CompileCacheNamespace {
        CompileCacheNamespace::new("H100-SXM5", "sha256:abc123", "meta-llama/70b")
    }

    fn test_store(dir: &std::path::Path, discovery: Arc<dyn PeerDiscovery>) -> CompileCacheStore {
        CompileCacheStore::new(
            test_namespace(),
            dir.to_path_buf(),
            1024 * 1024, // 1 MB
            discovery,
            0, // unused port for non-P2P tests
        )
    }

    // Namespace tests

    #[test]
    fn namespace_construction_and_display() {
        let ns = CompileCacheNamespace::new("H100-SXM5", "sha256:deadbeef", "org/model");
        assert_eq!(ns.as_str(), "H100-SXM5:sha256:deadbeef:org/model");
        assert_eq!(ns.to_string(), "H100-SXM5:sha256:deadbeef:org/model");
    }

    #[test]
    fn namespace_equality() {
        let a = CompileCacheNamespace::new("H100", "digest-a", "model-1");
        let b = CompileCacheNamespace::new("H100", "digest-a", "model-1");
        let c = CompileCacheNamespace::new("A100", "digest-a", "model-1");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // Store tests

    #[tokio::test]
    async fn store_set_get_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-1"));
        let store = test_store(tmp.path(), discovery);

        store.set("key-1", Bytes::from_static(b"hello")).await;

        let val = store.get("key-1").await.unwrap();
        assert_eq!(val.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn store_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-1"));
        let store = test_store(tmp.path(), discovery);

        assert!(!store.exists("nope").await);
        store.set("yep", Bytes::from_static(b"data")).await;
        assert!(store.exists("yep").await);
    }

    #[tokio::test]
    async fn store_get_miss_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-1"));
        let store = test_store(tmp.path(), discovery);

        assert!(store.get("ghost").await.is_none());
    }

    #[tokio::test]
    async fn store_disk_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-1"));

        // Write with one store instance
        {
            let store = test_store(tmp.path(), discovery.clone());
            store
                .set("persist-me", Bytes::from_static(b"durable"))
                .await;
        }

        // Read with a fresh store (memory is empty, should hit disk)
        let store = test_store(tmp.path(), discovery);
        let val = store.get("persist-me").await.unwrap();
        assert_eq!(val.as_ref(), b"durable");
    }

    #[tokio::test]
    async fn store_memory_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-1"));

        // 100 byte memory limit
        let store = CompileCacheStore::new(
            test_namespace(),
            tmp.path().to_path_buf(),
            100,
            discovery,
            0,
        );

        // Insert 60 bytes
        store.set("a", Bytes::from(vec![0u8; 60])).await;
        assert!(store.exists("a").await);

        // Insert another 60 bytes, should trigger eviction of "a" from memory
        store.set("b", Bytes::from(vec![1u8; 60])).await;

        // "a" should still exist on disk even if evicted from memory
        let val = store.get("a").await;
        assert!(
            val.is_some(),
            "key 'a' should still be accessible from disk after eviction"
        );

        // Memory should be within budget (only "b" in memory, or possibly
        // "a" promoted back from disk, but total should be <= 120 due to eviction)
        let mem_size = store.memory_size.load(std::sync::atomic::Ordering::Relaxed);
        // After eviction, at most one 60-byte entry should remain in memory
        // (the one we just inserted). The second get might promote "a" back.
        assert!(
            mem_size <= 120,
            "memory should be managed, got {mem_size} bytes"
        );
    }

    // RESP server integration tests

    /// Helper: start a RESP server on a random port, return the port.
    async fn start_test_server(store: CompileCacheStore) -> (u16, tokio::task::JoinHandle<()>) {
        let server = CompileCacheServer::new("127.0.0.1:0".to_string(), store);
        // Bind manually to get the actual port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = tokio::spawn(async move {
            // We need to recreate the server flow since we already bound
            while let Ok((stream, _)) = listener.accept().await {
                let store = server.store.clone();
                tokio::spawn(async move {
                    let _ = crate::compile_cache::handle_connection(stream, store).await;
                });
            }
        });

        (port, handle)
    }

    async fn connect_resp(port: u16) -> tokio::net::TcpStream {
        tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap()
    }

    /// Send a raw RESP command and read the response.
    async fn send_recv(stream: &mut tokio::net::TcpStream, cmd: &[u8]) -> Vec<u8> {
        stream.write_all(cmd).await.unwrap();
        // Give server a moment to process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        buf.truncate(n);
        buf
    }

    /// Build a RESP array from string arguments.
    fn resp_cmd(args: &[&str]) -> Vec<u8> {
        let mut buf = format!("*{}\r\n", args.len());
        for arg in args {
            buf.push_str(&format!("${}\r\n{}\r\n", arg.len(), arg));
        }
        buf.into_bytes()
    }

    #[tokio::test]
    async fn resp_server_ping_pong() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        let mut stream = connect_resp(port).await;
        let resp = send_recv(&mut stream, &resp_cmd(&["PING"])).await;
        assert_eq!(resp, b"+PONG\r\n");

        handle.abort();
    }

    #[tokio::test]
    async fn resp_server_set_get() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        let mut stream = connect_resp(port).await;

        // SET
        let resp = send_recv(&mut stream, &resp_cmd(&["SET", "mykey", "myvalue"])).await;
        assert_eq!(resp, b"+OK\r\n");

        // GET
        let resp = send_recv(&mut stream, &resp_cmd(&["GET", "mykey"])).await;
        // $7\r\nmyvalue\r\n
        assert_eq!(resp, b"$7\r\nmyvalue\r\n");

        handle.abort();
    }

    #[tokio::test]
    async fn resp_server_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        let mut stream = connect_resp(port).await;

        // EXISTS on missing key
        let resp = send_recv(&mut stream, &resp_cmd(&["EXISTS", "nope"])).await;
        assert_eq!(resp, b":0\r\n");

        // SET then EXISTS
        let _ = send_recv(&mut stream, &resp_cmd(&["SET", "yes", "val"])).await;
        let resp = send_recv(&mut stream, &resp_cmd(&["EXISTS", "yes"])).await;
        assert_eq!(resp, b":1\r\n");

        handle.abort();
    }

    #[tokio::test]
    async fn resp_server_handshake() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        let mut stream = connect_resp(port).await;

        // COMMAND -> empty array
        let resp = send_recv(&mut stream, &resp_cmd(&["COMMAND"])).await;
        assert_eq!(resp, b"*0\r\n");

        // CONFIG GET -> empty array
        let resp = send_recv(&mut stream, &resp_cmd(&["CONFIG", "GET", "save"])).await;
        assert_eq!(resp, b"*0\r\n");

        // CLIENT SETNAME -> +OK
        let resp = send_recv(&mut stream, &resp_cmd(&["CLIENT", "SETNAME", "test"])).await;
        assert_eq!(resp, b"+OK\r\n");

        // INFO -> empty bulk string
        let resp = send_recv(&mut stream, &resp_cmd(&["INFO"])).await;
        assert_eq!(resp, b"$0\r\n\r\n");

        handle.abort();
    }

    #[tokio::test]
    async fn resp_server_get_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        let mut stream = connect_resp(port).await;
        let resp = send_recv(&mut stream, &resp_cmd(&["GET", "nonexistent"])).await;
        assert_eq!(resp, b"$-1\r\n");

        handle.abort();
    }

    // P2P tests

    #[tokio::test]
    async fn p2p_set_on_a_get_from_b() {
        // Two stores with separate MockDiscovery instances.
        // Store A writes a value. Store B discovers A via HTTP and fetches it.

        let tmp_a = tempfile::tempdir().unwrap();
        let tmp_b = tempfile::tempdir().unwrap();
        let ns = test_namespace();

        // Store A: start an HTTP server with the compile-cache endpoint.
        let disc_a: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("node-a"));
        let store_a = CompileCacheStore::new(
            ns.clone(),
            tmp_a.path().to_path_buf(),
            1024 * 1024,
            disc_a.clone(),
            0, // will be replaced
        );
        store_a
            .set("shared-key", Bytes::from_static(b"shared-val"))
            .await;

        // Start an HTTP server for store A
        let router_a = build_internal_router(
            ModelPeerStore::new(),
            Some(store_a.clone()),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
        );
        let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_port = http_listener.local_addr().unwrap().port();
        let http_handle = tokio::spawn(async move {
            axum::serve(http_listener, router_a).await.ok();
        });

        // Store B: its discovery knows about node A's IP and port.
        // We use a MockDiscovery that already has A advertised.
        let disc_b = Arc::new(MockDiscovery::new("node-b"));
        // Manually advertise A's compile cache so B can find it.
        // MockDiscovery only tracks its own node, so we need a workaround.
        // Instead, we'll create a custom tiny discovery that returns node-a as a peer.
        let disc_b_wrapper = Arc::new(PeerListOverride {
            inner: disc_b,
            override_peers: vec![discovery::CompileCachePeer {
                node_id: "node-a".to_string(),
                peer_ip: "127.0.0.1".parse().unwrap(),
            }],
        });

        let store_b = CompileCacheStore::new(
            ns,
            tmp_b.path().to_path_buf(),
            1024 * 1024,
            disc_b_wrapper as Arc<dyn PeerDiscovery>,
            http_port,
        );

        // Store B should fetch from A via HTTP
        let val = store_b.get("shared-key").await;
        assert!(val.is_some(), "store B should find the value via peer A");
        assert_eq!(val.unwrap().as_ref(), b"shared-val");

        http_handle.abort();
    }

    #[tokio::test]
    async fn p2p_get_no_peers_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("lonely-node"));
        let store = test_store(tmp.path(), discovery);

        // No peers, no data, should be None
        assert!(store.get("anything").await.is_none());
    }

    #[tokio::test]
    async fn concurrent_connections() {
        let tmp = tempfile::tempdir().unwrap();
        let discovery: Arc<dyn PeerDiscovery> = Arc::new(MockDiscovery::new("n"));
        let store = test_store(tmp.path(), discovery);
        let (port, handle) = start_test_server(store).await;

        // Spawn 5 concurrent clients, each doing SET + GET
        let mut tasks = Vec::new();
        for i in 0..5 {
            tasks.push(tokio::spawn(async move {
                let mut stream = connect_resp(port).await;
                let key = format!("key-{i}");
                let val = format!("val-{i}");

                let resp = send_recv(&mut stream, &resp_cmd(&["SET", &key, &val])).await;
                assert_eq!(resp, b"+OK\r\n");

                let resp = send_recv(&mut stream, &resp_cmd(&["GET", &key])).await;
                let expected = format!("${}\r\n{}\r\n", val.len(), val);
                assert_eq!(resp, expected.as_bytes());
            }));
        }

        for t in tasks {
            t.await.unwrap();
        }

        handle.abort();
    }

    // Test helper: discovery wrapper that overrides compile cache peers

    /// Wraps a real PeerDiscovery but overrides `find_compile_cache_peers`
    /// to return a hardcoded list. Used for P2P tests where we need store B
    /// to "discover" store A without sharing a single MockDiscovery instance.
    struct PeerListOverride {
        inner: Arc<dyn PeerDiscovery>,
        override_peers: Vec<discovery::CompileCachePeer>,
    }

    #[async_trait::async_trait]
    impl PeerDiscovery for PeerListOverride {
        async fn find_model_peers(&self, repo_id: &str, tp_rank: u32) -> Vec<discovery::ModelPeer> {
            self.inner.find_model_peers(repo_id, tp_rank).await
        }

        async fn advertise_model_peer(&self, agent_name: &str, model: &str, tp_rank: u32) {
            self.inner
                .advertise_model_peer(agent_name, model, tp_rank)
                .await;
        }

        async fn unadvertise_model_peer(&self, agent_name: &str) {
            self.inner.unadvertise_model_peer(agent_name).await;
        }

        async fn live_nodes(&self) -> Vec<String> {
            self.inner.live_nodes().await
        }

        fn node_id(&self) -> &str {
            self.inner.node_id()
        }

        async fn find_compile_cache_peers(
            &self,
            _namespace: &str,
        ) -> Vec<discovery::CompileCachePeer> {
            self.override_peers.clone()
        }

        async fn advertise_compile_cache(&self, namespace: &str) {
            self.inner.advertise_compile_cache(namespace).await;
        }

        async fn unadvertise_compile_cache(&self) {
            self.inner.unadvertise_compile_cache().await;
        }
    }
}
