//!   ┌─────────┐  try_acquire   ┌────────┐  renew loop  ┌────────┐
//!   │ Standby ├───────────────►│ Leader ├──────────────►│ Leader │
//!   └────┬────┘                └───┬────┘               └───┬────┘
//!        │                         │ lease expired           │
//!        │                         │ or step_down()          │
//!        │                         ▼                         │
//!        │                    ┌─────────┐                    │
//!        └────────────────────┤ Standby │◄───────────────────┘
//!                             └─────────┘

use std::sync::Arc;
use std::time::Duration;

use k8s_openapi::api::coordination::v1::Lease;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::MicroTime;
use kube::Client;
use kube::api::{Api, Patch, PatchParams, PostParams};
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

#[derive(Debug, thiserror::Error)]
pub enum LeaderError {
    #[error("kube error: {0}")]
    Kube(#[from] kube::Error),

    #[error("lease has no metadata")]
    MissingMetadata,
}

pub struct LeaderConfig {
    pub holder_id: String,
    pub lease_name: String,
    pub namespace: String,
    pub lease_ttl: Duration,
    pub renew_interval: Duration,
}

pub struct LeaderElector {
    config: LeaderConfig,
    client: Client,
    leader_tx: watch::Sender<bool>,
    leader_rx: watch::Receiver<bool>,
}

fn now_timestamp() -> k8s_openapi::jiff::Timestamp {
    k8s_openapi::jiff::Timestamp::now()
}

impl LeaderElector {
    pub fn new(client: Client, config: LeaderConfig) -> Self {
        let (leader_tx, leader_rx) = watch::channel(false);
        Self {
            config,
            client,
            leader_tx,
            leader_rx,
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.leader_rx.clone()
    }

    pub async fn run(self: Arc<Self>) {
        let leases: Api<Lease> = Api::namespaced(self.client.clone(), &self.config.namespace);

        loop {
            match self.try_acquire_or_renew(&leases).await {
                Ok(true) => {
                    if !*self.leader_tx.borrow() {
                        info!(
                            holder = %self.config.holder_id,
                            lease = %self.config.lease_name,
                            "acquired leadership"
                        );
                    }
                    self.leader_tx.send_replace(true);
                }
                Ok(false) => {
                    if *self.leader_tx.borrow() {
                        warn!(
                            holder = %self.config.holder_id,
                            "lost leadership"
                        );
                    }
                    self.leader_tx.send_replace(false);
                }
                Err(e) => {
                    error!(error = %e, "leader election error");
                    if *self.leader_tx.borrow() {
                        warn!("assuming leadership lost due to error");
                        self.leader_tx.send_replace(false);
                    }
                }
            }

            tokio::time::sleep(self.config.renew_interval).await;
        }
    }

    pub async fn step_down(&self) -> Result<(), LeaderError> {
        if !*self.leader_tx.borrow() {
            return Ok(());
        }

        let leases: Api<Lease> = Api::namespaced(self.client.clone(), &self.config.namespace);

        let patch = serde_json::json!({
            "spec": {
                "holderIdentity": serde_json::Value::Null,
                "leaseDurationSeconds": 1,
            }
        });

        leases
            .patch(
                &self.config.lease_name,
                &PatchParams::apply("layercast-leader"),
                &Patch::Merge(&patch),
            )
            .await?;

        self.leader_tx.send_replace(false);
        info!(holder = %self.config.holder_id, "stepped down from leadership");
        Ok(())
    }

    async fn try_acquire_or_renew(&self, leases: &Api<Lease>) -> Result<bool, LeaderError> {
        let now = now_timestamp();
        let ttl_secs = self.config.lease_ttl.as_secs() as i32;

        match leases.get_opt(&self.config.lease_name).await? {
            None => {
                let lease = self.build_lease(ttl_secs, now);
                match leases.create(&PostParams::default(), &lease).await {
                    Ok(_) => Ok(true),
                    Err(kube::Error::Api(e)) if e.code == 409 => {
                        debug!("lease creation conflict, will retry");
                        Ok(false)
                    }
                    Err(e) => Err(e.into()),
                }
            }
            Some(existing) => {
                let spec = existing.spec.as_ref();
                let holder = spec.and_then(|s| s.holder_identity.as_deref());
                let renew_time = spec.and_then(|s| s.renew_time.as_ref());
                let duration = spec.and_then(|s| s.lease_duration_seconds);

                let expired = match (renew_time, duration) {
                    (Some(MicroTime(t)), Some(d)) => {
                        let expires_at = *t + Duration::from_secs(d as u64);
                        now > expires_at
                    }
                    _ => true,
                };

                let we_hold_it = holder == Some(&self.config.holder_id);

                if we_hold_it {
                    self.renew_lease(leases, ttl_secs, now).await
                } else if expired {
                    self.acquire_lease(leases, ttl_secs, now).await
                } else {
                    debug!(
                        holder = holder.unwrap_or("<none>"),
                        "lease held by another instance"
                    );
                    Ok(false)
                }
            }
        }
    }

    fn build_lease(&self, ttl_secs: i32, now: k8s_openapi::jiff::Timestamp) -> Lease {
        Lease {
            metadata: kube::api::ObjectMeta {
                name: Some(self.config.lease_name.clone()),
                namespace: Some(self.config.namespace.clone()),
                ..Default::default()
            },
            spec: Some(k8s_openapi::api::coordination::v1::LeaseSpec {
                holder_identity: Some(self.config.holder_id.clone()),
                lease_duration_seconds: Some(ttl_secs),
                acquire_time: Some(MicroTime(now)),
                renew_time: Some(MicroTime(now)),
                lease_transitions: Some(0),
                ..Default::default()
            }),
        }
    }

    async fn renew_lease(
        &self,
        leases: &Api<Lease>,
        ttl_secs: i32,
        now: k8s_openapi::jiff::Timestamp,
    ) -> Result<bool, LeaderError> {
        let patch = serde_json::json!({
            "spec": {
                "renewTime": MicroTime(now),
                "leaseDurationSeconds": ttl_secs,
            }
        });

        match leases
            .patch(
                &self.config.lease_name,
                &PatchParams::apply("layercast-leader"),
                &Patch::Merge(&patch),
            )
            .await
        {
            Ok(_) => {
                debug!("lease renewed");
                Ok(true)
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                warn!("lease renewal conflict, lost leadership");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn acquire_lease(
        &self,
        leases: &Api<Lease>,
        ttl_secs: i32,
        now: k8s_openapi::jiff::Timestamp,
    ) -> Result<bool, LeaderError> {
        let current = leases.get(&self.config.lease_name).await?;
        let rv = current
            .metadata
            .resource_version
            .as_deref()
            .ok_or(LeaderError::MissingMetadata)?;

        let transitions = current
            .spec
            .as_ref()
            .and_then(|s| s.lease_transitions)
            .unwrap_or(0);

        let patch = serde_json::json!({
            "metadata": {
                "resourceVersion": rv,
            },
            "spec": {
                "holderIdentity": self.config.holder_id,
                "leaseDurationSeconds": ttl_secs,
                "acquireTime": MicroTime(now),
                "renewTime": MicroTime(now),
                "leaseTransitions": transitions + 1,
            }
        });

        match leases
            .patch(
                &self.config.lease_name,
                &PatchParams::default(),
                &Patch::Merge(&patch),
            )
            .await
        {
            Ok(_) => {
                info!(
                    holder = %self.config.holder_id,
                    transitions = transitions + 1,
                    "acquired lease from expired holder"
                );
                Ok(true)
            }
            Err(kube::Error::Api(e)) if e.code == 409 => {
                debug!("lease acquisition conflict, another candidate won");
                Ok(false)
            }
            Err(e) => Err(e.into()),
        }
    }
}
