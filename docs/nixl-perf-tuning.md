# NIXL Transfer Performance Tuning

Tracked optimizations for GPUDirect RDMA weight transfers.

## Done

- [x] **Chunked VRAM registration**: Group tensors into N contiguous
      buffers (power-of-2 count, ~8GB target) and register those instead
      of 258 individual tensors. Cuts `register_memory()` from ~5.3s to
      ~200ms (8 pins vs 258). Branch: `feat/chunked-vram-registration`.

## Next

- [ ] **NUMA-aware process binding**: Wrap the vLLM entrypoint with
      `numactl --cpunodebind=N --membind=N` where N is the NUMA node of
      the assigned GPU. Ensures CPU threads, memory allocations, and the
      UCX worker all live on the same socket as the GPU and NIC.

      Discovery snippet:
      ```bash
      GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader \
        | head -1 | tr '[:upper:]' '[:lower:]' | sed 's/^0000://')
      GPU_NUMA=$(cat /sys/bus/pci/devices/$GPU_PCI/numa_node 2>/dev/null || echo 0)
      exec numactl --cpunodebind=$GPU_NUMA --membind=$GPU_NUMA "$@"
      ```

- [ ] **UCX nearest-NIC selection**: Set `UCX_MAX_RNDV_RAILS=1` on the
      vLLM container. Forces UCX to pick the closest NIC to the GPU
      based on PCIe/NUMA topology instead of round-robining across all
      available devices.

- [ ] **K8s topology manager check**: Verify CoreWeave nodes run
      `--topology-manager-policy=single-numa-node`. If set, the kubelet
      guarantees GPU + RDMA VF + CPU cores are co-located on one NUMA
      node. If not set, the numactl wrapper above is the fallback.

## Backlog

- [ ] **Multi-rail transfers**: Request multiple `rdma/ib` VFs and
      stripe transfers across them. Theoretically 2x throughput per
      GPU (two NICs per socket on H200 SXM). Tradeoff: claiming extra
      RDMA VFs for a single-GPU pod is wasteful in a shared cluster.

- [ ] **Adaptive coalesce threshold**: Currently `LAYERCAST_COALESCE_THRESHOLD`
      is 1MB. Profile whether raising it (e.g. 4MB) reduces per-transfer
      overhead for medium-sized tensors without hurting large tensor
      pipelining.

- [ ] **Transfer pipelining**: Overlap NIXL reads with `param.copy_()`
      scatter operations. Currently we wait for all transfers, then
      scatter. Could pipeline: transfer chunk N while scattering chunk
      N-1.

- [ ] **GDRCopy for small tensors**: Use GDRCopy (CPU-initiated GPU
      memory copies) for tensors below the coalesce threshold instead
      of RDMA. Lower latency for tiny norm/bias tensors.
