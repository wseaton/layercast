//!   ┌──────────────┐  RESP (SET/GET)  ┌──────────────────────┐
//!   │  torch.compile├────────────────►│  CompileCacheServer   │
//!   │  (any pod)    │                 └──────────┬───────────┘
//!   └──────────────┘                             │
//!                                     ┌──────────▼───────────┐
//!                                     │  CompileCacheStore    │
//!                                     │  ┌────────┐ ┌──────┐ │
//!                                     │  │ memory │ │ disk │ │
//!                                     │  └────────┘ └──────┘ │
//!                                     └──────────────────────┘

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::resp::{self, RespCommand, RespResponse};

#[derive(Debug, thiserror::Error)]
pub enum CompileCacheError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Default)]
pub struct CompileCacheStats {
    pub gets: AtomicUsize,
    pub mem_hits: AtomicUsize,
    pub disk_hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub sets: AtomicUsize,
}

#[derive(serde::Serialize)]
pub struct CompileCacheStatsSnapshot {
    pub gets: usize,
    pub mem_hits: usize,
    pub disk_hits: usize,
    pub misses: usize,
    pub sets: usize,
}

#[derive(Clone)]
pub struct CompileCacheStore {
    memory: Arc<RwLock<HashMap<String, Bytes>>>,
    memory_size: Arc<AtomicUsize>,
    max_memory: usize,
    disk_dir: PathBuf,
    stats: Arc<CompileCacheStats>,
}

impl CompileCacheStore {
    pub fn new(disk_dir: PathBuf, max_memory: usize) -> Self {
        if let Err(e) = std::fs::create_dir_all(&disk_dir) {
            warn!(path = %disk_dir.display(), error = %e, "failed to create compile cache disk dir");
        }

        Self {
            memory: Arc::new(RwLock::new(HashMap::new())),
            memory_size: Arc::new(AtomicUsize::new(0)),
            max_memory,
            disk_dir,
            stats: Arc::new(CompileCacheStats::default()),
        }
    }

    pub fn stats(&self) -> CompileCacheStatsSnapshot {
        CompileCacheStatsSnapshot {
            gets: self.stats.gets.load(Ordering::Relaxed),
            mem_hits: self.stats.mem_hits.load(Ordering::Relaxed),
            disk_hits: self.stats.disk_hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            sets: self.stats.sets.load(Ordering::Relaxed),
        }
    }

    pub async fn get(&self, key: &str) -> Option<Bytes> {
        self.stats.gets.fetch_add(1, Ordering::Relaxed);

        if let Some(val) = self.get_memory(key).await {
            self.stats.mem_hits.fetch_add(1, Ordering::Relaxed);
            return Some(val);
        }

        if let Some(val) = self.get_disk(key).await {
            self.stats.disk_hits.fetch_add(1, Ordering::Relaxed);
            self.set_memory(key, val.clone()).await;
            return Some(val);
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn set(&self, key: &str, value: Bytes) {
        self.stats.sets.fetch_add(1, Ordering::Relaxed);
        self.set_memory(key, value.clone()).await;
        self.write_disk(key, &value).await;
    }

    pub async fn exists(&self, key: &str) -> bool {
        {
            let mem = self.memory.read().await;
            if mem.contains_key(key) {
                return true;
            }
        }
        let path = self.disk_path(key);
        tokio::fs::metadata(&path).await.is_ok()
    }

    async fn get_memory(&self, key: &str) -> Option<Bytes> {
        self.memory.read().await.get(key).cloned()
    }

    async fn set_memory(&self, key: &str, value: Bytes) {
        let value_len = value.len();
        let mut mem = self.memory.write().await;

        if let Some(old) = mem.get(key) {
            self.memory_size.fetch_sub(old.len(), Ordering::Relaxed);
        }

        mem.insert(key.to_string(), value);
        self.memory_size.fetch_add(value_len, Ordering::Relaxed);

        while self.memory_size.load(Ordering::Relaxed) > self.max_memory && mem.len() > 1 {
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
            warn!(key, path = %path.display(), error = %e, "compile cache: failed to write to disk");
        }
    }

    fn disk_path(&self, key: &str) -> PathBuf {
        let key_hash = hash_string(key);
        self.disk_dir.join(key_hash)
    }
}

pub struct CompileCacheServer {
    addr: String,
    store: CompileCacheStore,
}

impl CompileCacheServer {
    pub fn new(addr: String, store: CompileCacheStore) -> Self {
        Self { addr, store }
    }

    pub async fn start(self) -> Result<JoinHandle<()>, CompileCacheError> {
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
) -> Result<(), CompileCacheError> {
    let mut buf = BytesMut::with_capacity(4096);
    let mut out = BytesMut::with_capacity(1024);

    loop {
        let n = stream.read_buf(&mut buf).await?;
        if n == 0 {
            return Ok(());
        }

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
            _ => RespResponse::EmptyArray,
        },
    }
}

fn hash_string(s: &str) -> String {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::compile_cache::CompileCacheStore;

    fn test_store(dir: &std::path::Path) -> CompileCacheStore {
        CompileCacheStore::new(dir.to_path_buf(), 1024 * 1024)
    }

    #[tokio::test]
    async fn store_set_get_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());

        store.set("key-1", Bytes::from_static(b"hello")).await;
        let val = store.get("key-1").await.unwrap();
        assert_eq!(val.as_ref(), b"hello");
    }

    #[tokio::test]
    async fn store_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());

        assert!(!store.exists("nope").await);
        store.set("yep", Bytes::from_static(b"data")).await;
        assert!(store.exists("yep").await);
    }

    #[tokio::test]
    async fn store_get_miss_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());
        assert!(store.get("ghost").await.is_none());
    }

    #[tokio::test]
    async fn store_disk_persistence() {
        let tmp = tempfile::tempdir().unwrap();

        {
            let store = test_store(tmp.path());
            store
                .set("persist-me", Bytes::from_static(b"durable"))
                .await;
        }

        let store = test_store(tmp.path());
        let val = store.get("persist-me").await.unwrap();
        assert_eq!(val.as_ref(), b"durable");
    }

    #[tokio::test]
    async fn store_memory_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CompileCacheStore::new(tmp.path().to_path_buf(), 100);

        store.set("a", Bytes::from(vec![0u8; 60])).await;
        assert!(store.exists("a").await);

        store.set("b", Bytes::from(vec![1u8; 60])).await;

        // "a" should still be accessible from disk
        let val = store.get("a").await;
        assert!(val.is_some());
    }

    async fn start_test_server(store: CompileCacheStore) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let store = store.clone();
                tokio::spawn(async move {
                    let _ = crate::compile_cache::handle_connection(stream, store).await;
                });
            }
        });

        (port, handle)
    }

    fn resp_cmd(args: &[&str]) -> Vec<u8> {
        let mut buf = format!("*{}\r\n", args.len());
        for arg in args {
            buf.push_str(&format!("${}\r\n{}\r\n", arg.len(), arg));
        }
        buf.into_bytes()
    }

    async fn send_recv(stream: &mut tokio::net::TcpStream, cmd: &[u8]) -> Vec<u8> {
        stream.write_all(cmd).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await.unwrap();
        buf.truncate(n);
        buf
    }

    #[tokio::test]
    async fn resp_server_ping_pong() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());
        let (port, handle) = start_test_server(store).await;

        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();
        let resp = send_recv(&mut stream, &resp_cmd(&["PING"])).await;
        assert_eq!(resp, b"+PONG\r\n");

        handle.abort();
    }

    #[tokio::test]
    async fn resp_server_set_get() {
        let tmp = tempfile::tempdir().unwrap();
        let store = test_store(tmp.path());
        let (port, handle) = start_test_server(store).await;

        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        let resp = send_recv(&mut stream, &resp_cmd(&["SET", "mykey", "myvalue"])).await;
        assert_eq!(resp, b"+OK\r\n");

        let resp = send_recv(&mut stream, &resp_cmd(&["GET", "mykey"])).await;
        assert_eq!(resp, b"$7\r\nmyvalue\r\n");

        handle.abort();
    }
}
