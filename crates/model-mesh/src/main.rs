use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};
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

    // start k8s discovery backend
    let k8s_config = discovery::K8sDiscoveryConfig {
        pod_name: config.pod_name.clone(),
        pod_namespace: config.pod_namespace.clone(),
        node_name: config.node_name.clone(),
        pod_ip: config.pod_ip.clone(),
        nixl_control_port: config.nixl_control_port,
    };
    let discovery: Arc<dyn discovery::PeerDiscovery> =
        Arc::new(discovery::K8sDiscovery::start(k8s_config).await?);

    let model_peer_store = ModelPeerStore::new();
    let file_cache = ipc::FileListCache::new();

    let http_port = config.http_port();

    // Build HF Hub API client (picks up HF_TOKEN automatically)
    let hf_api = hf_hub::api::tokio::ApiBuilder::new()
        .with_endpoint(config.hf_upstream.clone())
        .build()?;

    // start IPC server for vLLM plugin communication
    let ipc_server = Arc::new(ipc::IpcServer::new(
        config.ipc_socket_path.clone(),
        Arc::clone(&discovery),
        hf_api.clone(),
        model_peer_store.clone(),
        http_port,
        file_cache.clone(),
        Duration::from_secs(config.peer_discovery_timeout),
    ));
    // if MODEL_NAME is set, pre-fetch the file list + weight map
    // from HF API so the first PrepareModel is a cache hit.
    if let Some(ref model_name) = config.model_name {
        let model = model_name.clone();
        let ipc = Arc::clone(&ipc_server);
        tokio::spawn(async move {
            info!(model = %model, "predictive prefetch: fetching file list + weight map");
            match ipc.get_model_info(&model, "main").await {
                Ok(info) => {
                    info!(
                        model = %model,
                        file_count = info.files.len(),
                        weight_map_entries = info.weight_map.len(),
                        "predictive prefetch: cached"
                    );
                }
                Err(e) => {
                    warn!(
                        model = %model,
                        error = %e,
                        "predictive prefetch: failed to fetch from HF Hub"
                    );
                }
            }
        });
    }

    let _ipc_handle = ipc_server.start()?;
    info!(path = %config.ipc_socket_path.display(), "IPC server listening");

    // Compile cache: optionally start the RESP server + store
    let compile_cache = if config.compile_cache_enabled {
        let gpu_product = match &config.gpu_product {
            Some(gpu) => gpu.clone(),
            None => match detect_gpu_product(&config.node_name).await {
                Ok(gpu) => {
                    info!(gpu_product = %gpu, "auto-detected GPU product from node labels");
                    gpu
                }
                Err(e) => {
                    warn!(error = %e, "failed to detect GPU product, using 'unknown'");
                    "unknown".to_string()
                }
            },
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

    let ready = Arc::new(AtomicBool::new(false));
    let router = hf_api::build_internal_router(model_peer_store, compile_cache, Arc::clone(&ready));

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    info!(listen = %config.listen_addr, "HTTP server listening");

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

/// Auto-detect GPU product from the `nvidia.com/gpu.product` Kubernetes node label.
async fn detect_gpu_product(node_name: &str) -> Result<String> {
    let client = kube::Client::try_default().await?;
    let nodes: kube::Api<k8s_openapi::api::core::v1::Node> = kube::Api::all(client);
    let node = nodes.get(node_name).await?;

    node.metadata
        .labels
        .as_ref()
        .and_then(|labels| labels.get("nvidia.com/gpu.product"))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("node label nvidia.com/gpu.product not found"))
}
