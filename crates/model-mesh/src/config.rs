use std::path::PathBuf;

use clap::Parser;

/// Model weight distribution daemon for Kubernetes.
///
/// Runs as a sidecar alongside vLLM, providing:
/// - IPC server for the vLLM Layercast plugin (NIXL metadata, file discovery)
/// - Internal HTTP API for peer-to-peer NIXL VRAM metadata exchange
#[derive(Parser, Debug)]
#[command(name = "model-mesh", version)]
pub struct ProxyConfig {
    /// Kubernetes node name.
    #[arg(long, env = "NODE_NAME", default_value = "localhost")]
    pub node_name: String,

    /// Address to listen for HTTP requests (NIXL metadata exchange, health).
    #[arg(long, env = "LISTEN_ADDR", default_value = "0.0.0.0:8081")]
    pub listen_addr: String,

    /// Upstream HuggingFace Hub URL (used by IPC for model file listing).
    #[arg(long, env = "HF_UPSTREAM", default_value = "https://huggingface.co")]
    pub hf_upstream: String,

    /// Pod name (downward API: metadata.name). Required.
    #[arg(long, env = "POD_NAME")]
    pub pod_name: String,

    /// Pod IP address (downward API: status.podIP). Required.
    #[arg(long, env = "POD_IP")]
    pub pod_ip: String,

    /// Pod namespace (downward API: metadata.namespace).
    #[arg(long, env = "POD_NAMESPACE", default_value = "layercast-system")]
    pub pod_namespace: String,

    /// TCP port for NIXL transport control channel (metadata exchange).
    #[arg(long, env = "NIXL_CONTROL_PORT", default_value_t = 7903)]
    pub nixl_control_port: u16,

    /// Unix socket path for vLLM plugin IPC.
    #[arg(
        long,
        env = "IPC_SOCKET_PATH",
        default_value = "/var/run/layercast/daemon.sock"
    )]
    pub ipc_socket_path: PathBuf,

    /// Model name for predictive prefetch (e.g. "Qwen/Qwen2.5-32B").
    /// When set, the sidecar pre-fetches the file list from HF API at boot
    /// so PrepareModel can skip the network round-trip.
    #[arg(long, env = "MODEL_NAME")]
    pub model_name: Option<String>,

    /// Enable the torch.compile P2P cache (RESP server + peer exchange).
    #[arg(long, env = "COMPILE_CACHE_ENABLED", default_value_t = false)]
    pub compile_cache_enabled: bool,

    /// Address for the RESP (Redis protocol) server for torch.compile cache.
    #[arg(long, env = "COMPILE_CACHE_ADDR", default_value = "127.0.0.1:6379")]
    pub compile_cache_addr: String,

    /// GPU product identifier (e.g. "NVIDIA-H100-SXM5-80GB").
    /// Auto-detected from node labels if not set.
    #[arg(long, env = "GPU_PRODUCT")]
    pub gpu_product: Option<String>,

    /// Container image digest for compile cache namespace.
    #[arg(long, env = "IMAGE_DIGEST")]
    pub image_digest: Option<String>,

    /// Directory for on-disk compile cache.
    #[arg(
        long,
        env = "COMPILE_CACHE_DIR",
        default_value = "/var/cache/layercast/compile-cache"
    )]
    pub compile_cache_dir: PathBuf,

    /// Maximum memory for compile cache (bytes).
    #[arg(long, env = "COMPILE_CACHE_MAX_MEMORY", default_value_t = 512 * 1024 * 1024)]
    pub compile_cache_max_memory: usize,

    /// How long (seconds) to poll for NIXL peers during PrepareModel before
    /// giving up and returning an empty peer list. 0 = single shot, no retry.
    /// Useful for consumers that start before the seed has advertised its CRD.
    #[arg(long, env = "PEER_DISCOVERY_TIMEOUT", default_value_t = 120)]
    pub peer_discovery_timeout: u64,
}
