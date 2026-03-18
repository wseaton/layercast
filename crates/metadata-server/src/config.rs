use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "metadata-server", version)]
pub struct ServerConfig {
    #[arg(long, env = "POD_NAME")]
    pub pod_name: String,

    #[arg(long, env = "POD_NAMESPACE", default_value = "layercast-system")]
    pub pod_namespace: String,

    #[arg(long, env = "GRPC_ADDR", default_value = "0.0.0.0:50051")]
    pub grpc_addr: String,

    #[arg(long, env = "HTTP_ADDR", default_value = "0.0.0.0:8081")]
    pub http_addr: String,

    #[arg(long, env = "HF_UPSTREAM", default_value = "https://huggingface.co")]
    pub hf_upstream: String,

    #[arg(long, env = "COMPILE_CACHE_ENABLED", default_value_t = false)]
    pub compile_cache_enabled: bool,

    #[arg(long, env = "COMPILE_CACHE_ADDR", default_value = "0.0.0.0:6379")]
    pub compile_cache_addr: String,

    #[arg(
        long,
        env = "COMPILE_CACHE_DIR",
        default_value = "/var/cache/layercast/compile-cache"
    )]
    pub compile_cache_dir: PathBuf,

    /// Default 2GB.
    #[arg(long, env = "COMPILE_CACHE_MAX_MEMORY", default_value_t = 2 * 1024 * 1024 * 1024)]
    pub compile_cache_max_memory: usize,

    #[arg(long, env = "LEASE_NAME", default_value = "layercast-leader")]
    pub lease_name: String,

    #[arg(long, env = "LEASE_TTL", default_value_t = 15)]
    pub lease_ttl: u64,

    #[arg(long, env = "LEASE_RENEW_INTERVAL", default_value_t = 5)]
    pub lease_renew_interval: u64,

    #[arg(long, env = "PEER_DISCOVERY_TIMEOUT", default_value_t = 120)]
    pub peer_discovery_timeout: u64,

    /// Pods matching this selector are tracked for crash detection.
    #[arg(
        long,
        env = "POD_LABEL_SELECTOR",
        default_value = "layercast.io/managed=true"
    )]
    pub pod_label_selector: String,

    #[arg(long, env = "STATE_CONFIGMAP", default_value = "layercast-state")]
    pub state_configmap: String,
}
