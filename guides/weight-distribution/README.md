# Well-Lit Path: Weight Distribution

Accelerate LLM cold starts by distributing model weights directly between GPU VRAM via NIXL GPUDirect RDMA, bypassing disk and CPU entirely.

## Overview

A **seed** pod downloads the model once (from HuggingFace or a shared filesystem) and holds it in GPU memory. Every subsequent **consumer** pod pulls weights straight from the seed's VRAM at ~8 GB/s over InfiniBand. An optional torch.compile cache propagation layer means the second consumer onwards skips codegen entirely.

```
  ┌──────────────────┐         ┌──────────────────┐
  │    Seed Pod       │         │  Consumer Pod     │
  │                   │         │                   │
  │  vLLM + Plugin ──IPC── model-mesh              │
  │  (loads from HF)  │         │  vLLM + Plugin ──IPC── model-mesh
  │                   │         │  (loads via NIXL)  │
  │  GPU VRAM ════════════RDMA═══════════ GPU VRAM  │
  └──────────────────┘         └──────────────────┘
           │                            │
           └──── PodCache CRD ──────────┘
                (peer discovery)
```

Layercast runs as a **sidecar** alongside vLLM. The two communicate over a Unix domain socket. The sidecar handles peer discovery (via a Kubernetes CRD), NIXL metadata exchange (via HTTP), and compile cache sharing (via a RESP2 shim).

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
> Without InfiniBand, layercast falls back to the standard vLLM loading path (HuggingFace download or shared filesystem). The sidecar is a no-op, no RDMA hardware means no harm.

## Prerequisites

- A Kubernetes cluster with GPU nodes and InfiniBand networking
- `kubectl` configured with cluster access
- Container images available (see [Images](#images) below)

### Images

Two container images, both pushed to `ghcr.io/wseaton/layercast`:

| Image | Tag Pattern | Base | Purpose |
|-------|-------------|------|---------|
| **model-mesh** | `model-mesh-{sha\|branch\|vN}` | UBI9 | Rust sidecar daemon |
| **vllm-plugin** | `vllm-plugin-{sha\|branch\|vN}` | `ghcr.io/llm-d/llm-d-cuda` | vLLM with layercast plugin pre-installed |

### Cluster Resources

Apply the PodCache CRD and RBAC before deploying any pods:

```bash
# CRD (cluster-scoped, apply once)
kubectl apply -f https://raw.githubusercontent.com/wseaton/layercast/main/deploy/crds/podcache.yaml
```

```bash
# RBAC (namespace-scoped, apply per namespace)
export NAMESPACE=llm-d

kubectl apply -n $NAMESPACE -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: model-mesh
  labels:
    app.kubernetes.io/part-of: layercast
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: model-mesh
  labels:
    app.kubernetes.io/part-of: layercast
rules:
  - apiGroups: ["layercast.io"]
    resources: ["podcaches", "podcaches/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: model-mesh
  labels:
    app.kubernetes.io/part-of: layercast
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: model-mesh
subjects:
  - kind: ServiceAccount
    name: model-mesh
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: model-mesh-node-reader
  labels:
    app.kubernetes.io/part-of: layercast
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: model-mesh-node-reader
  labels:
    app.kubernetes.io/part-of: layercast
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: model-mesh-node-reader
subjects:
  - kind: ServiceAccount
    name: model-mesh
    namespace: llm-d
EOF
```

> [!IMPORTANT]
> Update the `namespace` field in the `ClusterRoleBinding` subject to match your deployment namespace.

## Installation

There are two deployment paths: injecting layercast into an existing llm-d helm deployment (recommended), or deploying standalone with kustomize.

---

### Option A: Inject into llm-d Helm Chart (Recommended)

The [`llm-d-modelservice`](https://github.com/llm-d-incubation/llm-d-modelservice) chart supports additional containers and volumes via its `decode.containers[]` and `decode.volumes[]` fields. Layercast plugs in as a values override file, no chart fork needed.

A tested values file is provided at [`layercast-values.yaml`](layercast-values.yaml). The key pieces:

- `serviceAccountOverride: model-mesh` points the pod at the RBAC created above
- `decode.containers[0]` (vllm) uses `modelCommand: vllmServe` with `--load-format=layercast` and the IPC socket env vars
- `decode.containers[1]` (model-mesh) uses `modelCommand: custom` with `command: ["model-mesh"]` and all sidecar env vars
- `decode.volumes` adds the shared IPC socket, `/dev/shm`, and compile cache emptyDirs

> [!IMPORTANT]
> The chart auto-injects `nvidia.com/gpu` into every container based on `parallelism.tensor`. The sidecar doesn't need a GPU, so its resources explicitly set `nvidia.com/gpu: "0"` to suppress this.

Copy the file into your working directory, then adjust `modelArtifacts.name` and `MODEL_NAME` to match your model.

Deploy with helm:

```bash
export NAMESPACE=llm-d

# Install the chart with the layercast overlay
helm install my-model llm-d-modelservice/llm-d-modelservice \
  -n $NAMESPACE \
  -f layercast-values.yaml
```

#### Seed vs Consumer Configuration

The overlay above works for both seed and consumer pods. The key difference is the `PEER_DISCOVERY_TIMEOUT` env var on the sidecar:

| Role | `PEER_DISCOVERY_TIMEOUT` | Behavior |
|------|--------------------------|----------|
| **Seed** | `0` | Skips peer discovery, loads from HF/filesystem immediately |
| **Consumer** | `120` (default) | Polls PodCache CRD for peers, loads via NIXL when found |

For a seed deployment, uncomment the `PEER_DISCOVERY_TIMEOUT` line in the values file and set it to `"0"`. Consumer deployments use the default and need no change.

> [!TIP]
> In practice, maintain two values files (e.g. `layercast-seed-values.yaml` and `layercast-consumer-values.yaml`) that differ only in `PEER_DISCOVERY_TIMEOUT`. The seed file uncomments the line and sets it to `"0"`.

---

### Option B: Standalone Kustomize

For deployments outside the llm-d ecosystem, use the kustomize overlay directly:

```bash
export NAMESPACE=layercast

# Edit deploy/nixl-e2e/kustomization.yaml to set your image tags
kubectl apply -k deploy/nixl-e2e/
```

This creates a seed deployment + consumer deployment with all RBAC, CRD, and services included.

## Verify the Installation

```bash
export NAMESPACE=llm-d   # or layercast, depending on your setup

# Check pods are running
kubectl get pods -n $NAMESPACE -l app.kubernetes.io/part-of=layercast
```

Expected output (two pods, each with 2/2 containers ready):

```
NAME                             READY   STATUS    RESTARTS   AGE
vllm-seed-7b9f4d6c8-x2k9p       2/2     Running   0          3m
vllm-consumer-5c8d7e4f2-m4n7q   2/2     Running   0          2m
```

Check PodCache CRDs are being created:

```bash
kubectl get podcaches -n $NAMESPACE
```

Expected:

```
NAME                              NODE        IP
pod-vllm-seed-7b9f4d6c8-x2k9p    gpu-node-1  10.244.1.5
pod-vllm-consumer-5c8d7e4f2-...   gpu-node-2  10.244.2.8
```

Verify the sidecar is healthy:

```bash
# Readiness (should return 200 once initialized)
kubectl exec -n $NAMESPACE deploy/vllm-seed -c model-mesh -- curl -s localhost:8081/healthz

# Check compile cache stats
kubectl exec -n $NAMESPACE deploy/vllm-consumer -c model-mesh -- curl -s localhost:8081/internal/compile-cache-stats
```

## Send a Test Request

Once both pods show `2/2 Running`, send a request to the consumer's vLLM endpoint:

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

The `deploy/benchmark/` directory contains kustomize overlays for comparative benchmarking:

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
export NAMESPACE=llm-d

# Helm
helm uninstall my-model -n $NAMESPACE

# Kustomize
kubectl delete -k deploy/nixl-e2e/

# CRD (cluster-wide, only if removing layercast entirely)
kubectl delete -f deploy/crds/podcache.yaml
```

## Configuration Reference

### model-mesh Sidecar

| Variable | Default | Description |
|----------|---------|-------------|
| `POD_NAME` | *(required)* | Pod name (K8s downward API) |
| `POD_IP` | *(required)* | Pod IP (K8s downward API) |
| `POD_NAMESPACE` | `layercast-system` | Pod namespace (K8s downward API) |
| `NODE_NAME` | `localhost` | Node name (K8s downward API) |
| `LISTEN_ADDR` | `0.0.0.0:8081` | HTTP server bind address |
| `IPC_SOCKET_PATH` | `/var/run/layercast/daemon.sock` | Unix socket for vLLM IPC |
| `HF_UPSTREAM` | `https://huggingface.co` | HuggingFace Hub URL |
| `NIXL_CONTROL_PORT` | `7903` | TCP port for NIXL metadata exchange |
| `MODEL_NAME` | *(none)* | Pre-fetch file list at boot |
| `PEER_DISCOVERY_TIMEOUT` | `120` | Seconds to poll for peers (0 = skip) |
| `COMPILE_CACHE_ENABLED` | `false` | Enable torch.compile P2P cache |
| `COMPILE_CACHE_ADDR` | `127.0.0.1:6379` | RESP2 shim listen address |
| `COMPILE_CACHE_DIR` | `/var/cache/layercast/compile-cache` | On-disk compile cache path |
| `COMPILE_CACHE_MAX_MEMORY` | `512MB` | Max in-memory compile cache |
| `GPU_PRODUCT` | *(auto-detected)* | GPU identifier |
| `IMAGE_DIGEST` | *(none)* | Image digest for compile cache namespace |

### vLLM Plugin

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYERCAST_SOCKET` | `/var/run/layercast/daemon.sock` | Path to sidecar IPC socket |
| `LAYERCAST_CHECKSUM` | `true` | Verify safetensor checksums |
| `LAYERCAST_COALESCE_THRESHOLD` | `1MB` | Coalesce NIXL transfers below this size |
| `LAYERCAST_NIXL_NUM_THREADS` | `4` | NIXL transfer thread count |
| `LAYERCAST_IB_SL` | `1` | InfiniBand service level |
