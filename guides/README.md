# Layercast Guides

A well-lit path is a documented, tested, and benchmarked deployment recipe. Each guide covers one concrete scenario end-to-end, from prerequisites to verification to cleanup. Where rough edges exist, they are called out explicitly.

## Available Paths

| Path | Description | Hardware |
|------|-------------|----------|
| [Weight Distribution](weight-distribution/) | P2P model weight loading via NIXL GPUDirect RDMA. One seed loads; every consumer pulls weights from GPU VRAM | NVIDIA GPUs with InfiniBand |
