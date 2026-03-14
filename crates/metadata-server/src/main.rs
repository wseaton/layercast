use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use metadata_server::compile_cache::{CompileCacheServer, CompileCacheStore};
use metadata_server::config::ServerConfig;
use metadata_server::grpc_service::LayercastGrpcServer;
use metadata_server::hf_api;
use metadata_server::leader::{LeaderConfig, LeaderElector};
use metadata_server::proto::layercast_service_server::LayercastServiceServer;
use metadata_server::state::{self, ServerState};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    let config = ServerConfig::parse();
    info!(
        pod_name = %config.pod_name,
        grpc_addr = %config.grpc_addr,
        http_addr = %config.http_addr,
        compile_cache = config.compile_cache_enabled,
        "starting layercast metadata server"
    );

    let k8s_client = kube::Client::try_default().await?;

    let leader_config = LeaderConfig {
        holder_id: config.pod_name.clone(),
        lease_name: config.lease_name.clone(),
        namespace: config.pod_namespace.clone(),
        lease_ttl: Duration::from_secs(config.lease_ttl),
        renew_interval: Duration::from_secs(config.lease_renew_interval),
    };
    let elector = Arc::new(LeaderElector::new(k8s_client.clone(), leader_config));
    let leader_rx = elector.subscribe();

    let elector_handle = Arc::clone(&elector);
    tokio::spawn(async move {
        elector_handle.run().await;
    });

    let server_state = Arc::new(ServerState::new());

    if let Some(persisted) =
        state::load_persisted_state(&k8s_client, &config.pod_namespace, &config.state_configmap)
            .await
    {
        server_state.restore_from_snapshot(persisted.peers).await;
    }

    let persist_state = Arc::clone(&server_state);
    let persist_client = k8s_client.clone();
    let persist_ns = config.pod_namespace.clone();
    let persist_cm = config.state_configmap.clone();
    let persist_id = config.pod_name.clone();
    tokio::spawn(async move {
        state::persist_loop(
            persist_state,
            persist_client,
            persist_ns,
            persist_cm,
            persist_id,
        )
        .await;
    });

    let watcher_state = Arc::clone(&server_state);
    let watcher_client = k8s_client.clone();
    let watcher_ns = config.pod_namespace.clone();
    let watcher_selector = config.pod_label_selector.clone();
    tokio::spawn(async move {
        loop {
            metadata_server::pod_watcher::watch_pods(
                watcher_client.clone(),
                watcher_ns.clone(),
                watcher_selector.clone(),
                Arc::clone(&watcher_state),
            )
            .await;
            warn!("pod watcher exited, restarting in 5s");
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    let hf_api = hf_hub::api::tokio::ApiBuilder::new()
        .with_endpoint(config.hf_upstream.clone())
        .build()?;

    let compile_cache = if config.compile_cache_enabled {
        let store = CompileCacheStore::new(
            config.compile_cache_dir.clone(),
            config.compile_cache_max_memory,
        );

        let server = CompileCacheServer::new(config.compile_cache_addr.clone(), store.clone());
        let _cache_handle = server.start().await?;

        info!(
            addr = %config.compile_cache_addr,
            max_memory = config.compile_cache_max_memory,
            "compile cache enabled"
        );

        Some(store)
    } else {
        None
    };

    let grpc_server = Arc::new(LayercastGrpcServer::new(
        Arc::clone(&server_state),
        hf_api,
        Duration::from_secs(config.peer_discovery_timeout),
    ));

    let grpc_addr = config.grpc_addr.parse()?;
    let _grpc_handle = tokio::spawn(async move {
        info!(%grpc_addr, "gRPC server listening");
        if let Err(e) = Server::builder()
            .add_service(LayercastServiceServer::from_arc(grpc_server))
            .serve(grpc_addr)
            .await
        {
            tracing::error!(error = %e, "gRPC server error");
        }
    });

    let health_router = hf_api::build_health_router(leader_rx, compile_cache);
    let http_listener = tokio::net::TcpListener::bind(&config.http_addr).await?;
    info!(listen = %config.http_addr, "HTTP health server listening");

    let _http_handle = tokio::spawn(async move {
        axum::serve(http_listener, health_router)
            .with_graceful_shutdown(async {
                tokio::signal::ctrl_c().await.ok();
            })
            .await
            .ok();
    });

    info!("metadata server ready");

    tokio::signal::ctrl_c().await?;
    info!("shutting down metadata server");

    if let Err(e) = elector.step_down().await {
        warn!(error = %e, "failed to step down from leadership");
    }

    Ok(())
}
