use std::sync::Arc;

use futures::TryStreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::Client;
use kube::api::Api;
use kube::runtime::watcher;
use tracing::{debug, error, info, warn};

use crate::state::ServerState;

pub async fn watch_pods(
    client: Client,
    namespace: String,
    label_selector: String,
    state: Arc<ServerState>,
) {
    let pods: Api<Pod> = Api::namespaced(client, &namespace);

    info!(
        namespace,
        label_selector, "starting pod watcher for crash detection"
    );

    let stream = watcher::watcher(pods, watcher::Config::default().labels(&label_selector));

    let mut stream = std::pin::pin!(stream);
    loop {
        match stream.try_next().await {
            Ok(Some(event)) => match event {
                watcher::Event::Delete(pod) => {
                    let pod_name = pod.metadata.name.as_deref().unwrap_or("<unknown>");
                    info!(pod_name, "pod deleted, cleaning up agents");
                    state.unregister_pod(pod_name).await;
                }
                watcher::Event::Apply(pod) => {
                    if let Some(status) = &pod.status
                        && let Some(phase) = &status.phase
                        && (phase == "Failed" || phase == "Succeeded")
                    {
                        let pod_name = pod.metadata.name.as_deref().unwrap_or("<unknown>");
                        debug!(pod_name, phase, "pod reached terminal phase, cleaning up");
                        state.unregister_pod(pod_name).await;
                    }
                }
                watcher::Event::InitApply(_) | watcher::Event::Init | watcher::Event::InitDone => {}
            },
            Ok(None) => {
                warn!("pod watcher stream ended, restarting");
                break;
            }
            Err(e) => {
                error!(error = %e, "pod watcher error");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}
