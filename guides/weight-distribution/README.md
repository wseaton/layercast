# Well-Lit Path: Weight Distribution

Accelerate LLM cold starts by distributing model weights directly between GPU VRAM via NIXL GPUDirect RDMA, bypassing disk and CPU entirely.

## Overview

A **seed** pod downloads the model once (from HuggingFace or a shared filesystem) and holds it in GPU memory. Every subsequent **consumer** pod pulls weights straight from the seed's VRAM at ~8 GB/s over InfiniBand. An optional torch.compile cache propagation layer means the second consumer onwards skips codegen entirely.

```
  ┌──────────────────┐         ┌──────────────────┐
  │    Seed Pod       │         │  Consumer Pod     │
  │                   │         │                   │
  │  vLLM + Plugin    │         │  vLLM + Plugin    │
  │  (loads from HF)  │         │  (loads via NIXL)  │
  │                   │         │                   │
  │  GPU VRAM ════════════RDMA═══════════ GPU VRAM  │
  └────────┬──────────┘         └────────┬──────────┘
           │          gRPC               │
           └──────── Metadata ───────────┘
                     Server
               (peer discovery,
                compile cache)
```

A central **metadata server** runs as its own Deployment (not a sidecar). vLLM pods communicate with it over gRPC. The server handles peer discovery, NIXL metadata exchange, compile cache sharing (RESP2), and automatic crash cleanup via a Pod watcher.

### Performance (Qwen2.5-32B, H100 SXM5, InfiniBand)

| Metric | Value |
|--------|-------|
| Transfer speed | 7.7 - 8.7 GB/s |
| Weight load (consumer, NIXL) | ~26s for 60GB model |
| Checksum verification | ~35ms |
| Compile cache savings | ~13s per consumer |

## Hardware Requirements

- **NVIDIA GPUs** with InfiniBand RDMA support (tested on H100 SXM5)
- **RDMA device plugin** exposing `rdma/ib` resources in the cluster
- Pods must be schedulable with `IPC_LOCK` capability (required for RDMA memory pinning)

> [!NOTE]
> Without InfiniBand, layercast falls back to the standard vLLM loading path (HuggingFace download or shared filesystem). The plugin is a no-op, no RDMA hardware means no harm.

## Prerequisites

- A Kubernetes cluster with GPU nodes and InfiniBand networking
- `kubectl` configured with cluster access
- Container images available (see [Images](#images) below)

### Images

Two container images, both pushed to `ghcr.io/wseaton/layercast`:

| Image | Tag Pattern | Base | Purpose |
|-------|-------------|------|---------|
| **metadata-server** | `metadata-server-{sha\|branch\|vN}` | UBI9 | Central metadata + compile cache server |
| **vllm-plugin** | `vllm-plugin-{sha\|branch\|vN}` | `ghcr.io/llm-d/llm-d-cuda` | vLLM with layercast plugin pre-installed |

### Deploy the Metadata Server

The metadata server must be running before any vLLM pods start. It uses Kubernetes Lease-based leader election (2 replicas for HA).

```bash
export NAMESPACE=layercast

# Deploy metadata server (includes RBAC, Deployment, Service)
kubectl apply -k deploy/metadata-server/
```

This creates:

- `metadata-server` ServiceAccount with Role for ConfigMap persistence, Pod watching, and Lease election
- 2-replica Deployment with leader election (only the leader passes readiness)
- `layercast-metadata-server` Service exposing gRPC (50051), HTTP (8081), and RESP2 (6379)

## Installation

There are two deployment paths: injecting layercast into an existing llm-d helm deployment (recommended), or deploying standalone with kustomize.

---

### Option A: Inject into llm-d Helm Chart (Recommended)

The [`llm-d-modelservice`](https://github.com/llm-d-incubation/llm-d-modelservice) chart supports additional containers and volumes via its `decode.containers[]` and `decode.volumes[]` fields. Layercast plugs in as a values override file, no chart fork needed.

A tested values file is provided at [`layercast-values.yaml`](layercast-values.yaml). The key pieces:

- `decode.containers[0]` (vllm) uses `modelCommand: vllmServe` with `--load-format=layercast`
- `LAYERCAST_SERVER_ADDR` points to the metadata server's gRPC endpoint
- `TORCHINDUCTOR_REDIS_HOST` points to the metadata server for compile cache
- Pod label `layercast.io/managed: "true"` enables automatic crash cleanup

Copy the file into your working directory, then adjust `modelArtifacts.name` to match your model.

Deploy with helm:

```bash
export NAMESPACE=layercast

# Install the chart with the layercast overlay
helm install my-model llm-d-modelservice/llm-d-modelservice \
  -n $NAMESPACE \
  -f layercast-values.yaml
```

#### Seed vs Consumer Configuration

The overlay works for both seed and consumer pods. The key difference is the `PEER_DISCOVERY_TIMEOUT` env var on the vLLM container:

| Role | `PEER_DISCOVERY_TIMEOUT` | Behavior |
|------|--------------------------|----------|
| **Seed** | `0` | Skips peer discovery, loads from HF/filesystem immediately |
| **Consumer** | unset (server default: 120s) | Metadata server polls for peers, loads via NIXL when found |

For a seed deployment, uncomment the `PEER_DISCOVERY_TIMEOUT` line in the values file and set it to `"0"`. Consumer deployments use the server default and need no change.

---

### Option B: Standalone Kustomize

For deployments outside the llm-d ecosystem, use the kustomize overlay directly:

```bash
export NAMESPACE=layercast

# Deploys metadata server + seed + consumer
kubectl apply -k deploy/nixl-e2e/
```

This creates the metadata server, a seed deployment, and a consumer deployment with all RBAC and services included.

## Verify the Installation

```bash
export NAMESPACE=layercast

# Check metadata server is running (1/2 pods ready, the leader)
kubectl get pods -n $NAMESPACE -l app=metadata-server

# Check vLLM pods
kubectl get pods -n $NAMESPACE -l app=vllm
```

Expected output:

```
NAME                               READY   STATUS    RESTARTS   AGE
metadata-server-7b9f4d6c8-abc12   1/1     Running   0          5m
metadata-server-7b9f4d6c8-def34   0/1     Running   0          5m    # standby
vllm-seed-5c8d7e4f2-x2k9p         1/1     Running   0          3m
vllm-consumer-5c8d7e4f2-m4n7q     1/1     Running   0          2m
```

Verify the metadata server is healthy:

```bash
# Readiness (returns 200 for the leader)
kubectl exec -n $NAMESPACE deploy/metadata-server -- curl -s localhost:8081/healthz

# Compile cache stats
kubectl exec -n $NAMESPACE deploy/metadata-server -- curl -s localhost:8081/internal/compile-cache-stats
```

## Send a Test Request

Once both vLLM pods show `1/1 Running`, send a request to the consumer:

```bash
kubectl port-forward -n $NAMESPACE deploy/vllm-consumer 8000:8000 &

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }' | jq .
```

## Benchmarking

The `deploy/benchmark/` directory contains kustomize overlays for comparative benchmarking. All require the metadata server to be deployed first (`kubectl apply -k deploy/metadata-server/`).

| Scenario | Description | Command |
|----------|-------------|---------|
| `hf-cold` | Cold HuggingFace download (baseline) | `kubectl apply -f deploy/benchmark/hf-cold.yaml` |
| `nfs-cached` | NFS-cached model load | `kubectl apply -f deploy/benchmark/nfs-cached.yaml` |
| `nixl-p2p` | NIXL P2P (seed + 1 consumer) | `kubectl apply -f deploy/benchmark/nixl-p2p.yaml` |
| `nixl-scaling` | NIXL + compile cache (seed + 2 consumers) | `kubectl apply -f deploy/benchmark/nixl-scaling.yaml` |

Measure time-to-ready by watching pod logs for the vLLM startup banner:

```bash
kubectl logs -n $NAMESPACE -f deploy/vllm-consumer -c vllm | grep -m1 "Uvicorn running"
```

## Cleanup

```bash
export NAMESPACE=layercast

# Helm
helm uninstall my-model -n $NAMESPACE

# Kustomize (full stack)
kubectl delete -k deploy/nixl-e2e/

# Metadata server only
kubectl delete -k deploy/metadata-server/
```

## Configuration Reference

### Metadata Server

| Variable | Default | Description |
|----------|---------|-------------|
| `POD_NAME` | *(required)* | Pod name (K8s downward API) |
| `POD_NAMESPACE` | `layercast-system` | Pod namespace (K8s downward API) |
| `GRPC_ADDR` | `0.0.0.0:50051` | gRPC server bind address |
| `HTTP_ADDR` | `0.0.0.0:8081` | HTTP health server bind address |
| `HF_UPSTREAM` | `https://huggingface.co` | HuggingFace Hub URL |
| `PEER_DISCOVERY_TIMEOUT` | `120` | Default seconds to poll for peers (overridable per-request) |
| `POD_LABEL_SELECTOR` | `layercast.io/managed=true` | Label selector for pod crash detection |
| `STATE_CONFIGMAP` | `layercast-state` | ConfigMap name for state persistence |
| `LEASE_NAME` | `layercast-leader` | Lease name for leader election |
| `LEASE_TTL` | `15` | Lease TTL in seconds |
| `LEASE_RENEW_INTERVAL` | `5` | Lease renewal interval in seconds |
| `COMPILE_CACHE_ENABLED` | `false` | Enable torch.compile RESP2 cache |
| `COMPILE_CACHE_ADDR` | `0.0.0.0:6379` | RESP2 shim listen address |
| `COMPILE_CACHE_DIR` | `/var/cache/layercast/compile-cache` | On-disk compile cache path |
| `COMPILE_CACHE_MAX_MEMORY` | `2GB` | Max in-memory compile cache |

### vLLM Plugin

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYERCAST_SERVER_ADDR` | `layercast-metadata-server:50051` | Metadata server gRPC address |
| `POD_NAME` | `$HOSTNAME` | Pod identity for server registration |
| `POD_IP` | *(required for NIXL)* | Pod IP (K8s downward API) |
| `PEER_DISCOVERY_TIMEOUT` | *(server default)* | Per-request override: seconds to wait for peers (0 = skip) |
| `LAYERCAST_CHECKSUM` | `true` | Verify safetensor checksums after transfer |
| `LAYERCAST_COALESCE_THRESHOLD` | `1MB` | Coalesce NIXL transfers below this size |
| `LAYERCAST_NIXL_NUM_THREADS` | `4` | NIXL transfer thread count |
| `LAYERCAST_IB_SL` | `1` | InfiniBand service level |
| `LAYERCAST_PARALLEL_PEER_XFER` | `1` | Overlap RDMA reads from multiple peers (0 = serialize) |
