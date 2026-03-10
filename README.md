# Layercast

Distributed model weight loading for Kubernetes. Accelerates LLM cold starts by transferring weights directly between GPU VRAM via NIXL GPUDirect RDMA, bypassing disk and CPU entirely.

## How it works

A seed pod downloads the model once (from HuggingFace or a shared filesystem). Every subsequent pod pulls weights straight from the seed's GPU memory at ~8 GB/s over InfiniBand. A torch.compile cache propagates between peers so the second consumer onwards skips codegen entirely.

```
Seed Pod                    Consumer Pods
┌──────────────────┐        ┌──────────────────┐
│ vLLM + Plugin    │        │ vLLM + Plugin    │
│   ↕ IPC          │        │   ↕ IPC          │
│ model-mesh       │        │ model-mesh       │
│   ↕ CRD          │        │   ↕ CRD          │
│ K8s NodeCache    │◄──────►│ K8s NodeCache    │
└──────────────────┘        └──────────────────┘
        │                           │
        └───── NIXL GPUDirect ──────┘
              (VRAM → VRAM RDMA)
```

## Weight loading cascade

The vLLM plugin tries each source in priority order, falling through on failure:

1. **NIXL GPUDirect** - VRAM-to-VRAM transfer from a peer pod (~8 GB/s over IB)
2. **vLLM default** - HuggingFace download or shared filesystem read

## Benchmark results (Qwen2.5-32B, CoreWeave)

| Scenario | Time-to-Ready | Weight Load | Method |
|----------|--------------|-------------|--------|
| NFS cached (VAST) | 119s | ~119s | NFS read |
| NIXL consumer | ~92s | 24s @ 8.6 GB/s | GPUDirect RDMA |
| NIXL + compile cache | ~81s | 25s @ 8.1 GB/s | GPUDirect + cache hits |

NIXL consumers load weights 3x faster than NFS. With compile cache hits (65/65), total TTR drops another 13%.

## Project structure

```
crates/
  discovery/       K8s peer discovery (NodeCache CRD, kube-rs reflectors)
  model-mesh/      Sidecar daemon (IPC server, HTTP metadata API, compile cache)
vllm-plugin/       vLLM plugin (model loader, NIXL agent, checksum verification)
deploy/
  benchmark/       Benchmark scenarios (NFS, NIXL scaling)
  nixl-e2e/        End-to-end NIXL deployment (kustomize)
```

| Crate | Description |
|-------|-------------|
| **discovery** | Peer discovery via K8s NodeCache CRD and kube-rs reflectors. Tracks model peers and compile cache namespaces. |
| **model-mesh** | IPC daemon for the vLLM plugin. HTTP server for peer metadata exchange, RESP2 shim for torch.compile P2P cache. |

## Sidecar HTTP API

The model-mesh sidecar exposes an HTTP server (default port 8081):

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Readiness probe (503 until init complete, then 200) |
| `GET /health` | Liveness check |
| `GET /internal/nixl-vram/:agent` | Fetch NIXL VRAM metadata for a peer |
| `GET /internal/compile-cache/:key` | Fetch compile cache entry from peer |
| `GET /internal/compile-cache-stats` | Compile cache hit/miss counters |

## IPC protocol

The vLLM plugin communicates with the sidecar over a Unix domain socket using length-prefixed msgpack. Three messages form a per-connection state machine:

```
    PrepareModel ──► Prepared{files, peers}
         │
    ModelLoaded  ──► Ok  (advertise to peers)
         │
    ModelUnloaded ──► Ok  (unadvertise, cleanup)
```

## Configuration

### model-mesh sidecar

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `0.0.0.0:8081` | HTTP server bind address |
| `IPC_SOCKET_PATH` | `/var/run/layercast/daemon.sock` | Unix socket for vLLM IPC |
| `HF_UPSTREAM` | `https://huggingface.co` | HuggingFace Hub URL (file listing) |
| `MODEL_NAME` | *(none)* | Pre-fetch file list at boot (predictive cache) |
| `PEER_DISCOVERY_TIMEOUT` | `120` | Seconds to poll for peers (0 = skip, for seeds) |
| `COMPILE_CACHE_ENABLED` | `false` | Enable torch.compile P2P cache |
| `COMPILE_CACHE_ADDR` | `127.0.0.1:6379` | RESP2 shim listen address |
| `POD_NAME` | *(required)* | From K8s downward API |
| `POD_IP` | *(required)* | From K8s downward API |
| `NODE_NAME` | *(required)* | From K8s downward API |

### vLLM plugin

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYERCAST_SOCKET` | `/var/run/layercast/daemon.sock` | Path to sidecar IPC socket |

## Quick start

```bash
# Build the sidecar
cargo build --release

# Install the vLLM plugin
pip install ./vllm-plugin

# Run vLLM with layercast
vllm serve --load-format layercast --model Qwen/Qwen2.5-32B
```

## Deploy to Kubernetes

```bash
# Apply the NIXL end-to-end setup (edit kustomization.yaml for your registry)
kubectl apply -k deploy/nixl-e2e/

# Run benchmarks
./scripts/benchmark.sh --skip-hf --skip-populate
```

## Development

```bash
# Format + lint
cargo fmt --all
cargo clippy --all --benches --tests --examples --all-features

# Run Rust tests
cargo test --all

# Run Python tests
cd vllm-plugin && uv run pytest tests/ -v

# Generate the NodeCache CRD YAML
cargo run -p discovery --bin crd-gen
```

## License

Apache-2.0
