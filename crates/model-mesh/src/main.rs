use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use model_mesh::compile_cache::{CompileCacheNamespace, CompileCacheServer, CompileCacheStore};
use model_mesh::config::ProxyConfig;
use model_mesh::hf_api;
use model_mesh::ipc;
use model_mesh::nixl_vram_store::ModelPeerStore;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    let config = ProxyConfig::parse();
    info!(
        node_name = %config.node_name,
        listen_addr = %config.listen_addr,
        hf_upstream = %config.hf_upstream,
        model_name = ?config.model_name,
        compile_cache = config.compile_cache_enabled,
        "starting model-mesh daemon"
    );

    // Start K8s discovery backend
    let k8s_config = discovery::K8sDiscoveryConfig {
        pod_name: config.pod_name.clone(),
        pod_namespace: config.pod_namespace.clone(),
        node_name: config.node_name.clone(),
        pod_ip: config.pod_ip.clone(),
        nixl_control_port: config.nixl_control_port,
    };
    let discovery: Arc<dyn discovery::PeerDiscovery> =
        Arc::new(discovery::K8sDiscovery::start(k8s_config).await?);

    // Model peer metadata store (shared between IPC server and HTTP API)
    let model_peer_store = ModelPeerStore::new();

    // File list cache for predictive prefetch
    let file_cache = ipc::FileListCache::new();

    // Start IPC server for vLLM plugin communication
    let ipc_server = Arc::new(ipc::IpcServer::new(
        config.ipc_socket_path.clone(),
        Arc::clone(&discovery),
        config.hf_upstream.clone(),
        model_peer_store.clone(),
        config.listen_addr.clone(),
        file_cache.clone(),
        Duration::from_secs(config.peer_discovery_timeout),
    ));
    let _ipc_handle = ipc_server.start()?;
    info!(path = %config.ipc_socket_path.display(), "IPC server listening");

    // Predictive prefetch: if MODEL_NAME is set, pre-fetch the file list
    // from HF API so the first PrepareModel is a cache hit.
    if let Some(ref model_name) = config.model_name {
        let model = model_name.clone();
        let hf_upstream = config.hf_upstream.clone();
        let cache = file_cache.clone();
        tokio::spawn(async move {
            info!(model = %model, "predictive prefetch: fetching file list");
            let url = format!("{}/api/models/{}/revision/main", hf_upstream, model);
            let client = reqwest::Client::new();
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        let files: Vec<String> = body
                            .get("siblings")
                            .and_then(|s| s.as_array())
                            .map(|siblings| {
                                siblings
                                    .iter()
                                    .filter_map(|s| s.get("rfilename").and_then(|f| f.as_str()))
                                    .filter(|f| f.ends_with(".safetensors"))
                                    .map(String::from)
                                    .collect()
                            })
                            .unwrap_or_default();
                        info!(
                            model = %model,
                            file_count = files.len(),
                            "predictive prefetch: file list cached"
                        );
                        cache.insert(&model, "main", files).await;
                    }
                }
                Ok(resp) => {
                    tracing::warn!(
                        model = %model,
                        status = %resp.status(),
                        "predictive prefetch: HF API returned non-success"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        model = %model,
                        error = %e,
                        "predictive prefetch: failed to reach HF API"
                    );
                }
            }
        });
    }

    // Compile cache: optionally start the RESP server + store
    let compile_cache = if config.compile_cache_enabled {
        let http_port = config
            .listen_addr
            .rsplit_once(':')
            .and_then(|(_, port)| port.parse::<u16>().ok())
            .unwrap_or(8081);

        let gpu_product = match &config.gpu_product {
            Some(gpu) => gpu.clone(),
            None => detect_gpu_product(&config.node_name).await,
        };

        let image_digest = config
            .image_digest
            .as_deref()
            .unwrap_or("unknown")
            .to_string();

        let model_name = config
            .model_name
            .as_deref()
            .unwrap_or("unknown")
            .to_string();

        let namespace = CompileCacheNamespace::new(&gpu_product, &image_digest, &model_name);
        info!(
            namespace = %namespace,
            addr = %config.compile_cache_addr,
            max_memory = config.compile_cache_max_memory,
            "compile cache enabled"
        );

        let store = CompileCacheStore::new(
            namespace,
            config.compile_cache_dir.clone(),
            config.compile_cache_max_memory,
            Arc::clone(&discovery),
            http_port,
        );

        let server = CompileCacheServer::new(config.compile_cache_addr.clone(), store.clone());
        let _cache_handle = server.start().await?;

        Some(store)
    } else {
        None
    };

    // Readiness flag: flipped to true once all initialization is done
    let ready = Arc::new(AtomicBool::new(false));

    // Build the internal HTTP API (NIXL metadata exchange + compile cache)
    let router = hf_api::build_internal_router(model_peer_store, compile_cache, Arc::clone(&ready));

    // Start HTTP server with graceful shutdown
    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    info!(listen = %config.listen_addr, "HTTP server listening");

    // All systems go: IPC socket bound, reflectors started, HTTP listener ready.
    ready.store(true, Ordering::Relaxed);
    info!("daemon ready");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
            info!("shutting down model-mesh");
        })
        .await?;

    Ok(())
}

/// Try to auto-detect GPU product from Kubernetes node labels.
/// Falls back to "unknown" if detection fails.
async fn detect_gpu_product(node_name: &str) -> String {
    match try_detect_gpu_product(node_name).await {
        Ok(gpu) => {
            info!(gpu_product = %gpu, "auto-detected GPU product from node labels");
            gpu
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to detect GPU product, using 'unknown'"
            );
            "unknown".to_string()
        }
    }
}

async fn try_detect_gpu_product(node_name: &str) -> Result<String> {
    let client = kube::Client::try_default().await?;
    let nodes: kube::Api<k8s_openapi::api::core::v1::Node> = kube::Api::all(client);
    let node = nodes.get(node_name).await?;

    let gpu = node
        .metadata
        .labels
        .as_ref()
        .and_then(|labels| labels.get("nvidia.com/gpu.product"))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("node label nvidia.com/gpu.product not found"))?;

    Ok(gpu)
}
