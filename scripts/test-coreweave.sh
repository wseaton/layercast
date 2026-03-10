#!/usr/bin/env bash
# End-to-end test for model-mesh on CoreWeave.
#
# Tests the full cascade: HF upstream -> pod-0, then pod-0 -> pod-1
# via peer transport (RDMA/NIXL/TCP).
#
# Usage:
#   ./scripts/test-coreweave.sh
#   MODEL=meta-llama/Llama-3.2-1B FILE=model.safetensors ./scripts/test-coreweave.sh

set -euo pipefail

# Override with KUBE_CONTEXT env var to target a different cluster.
CTX="${KUBE_CONTEXT:-coreweave-waldorf}"
NS="layercast"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
FILE="${FILE:-model-00001-of-00002.safetensors}"
REVISION="${REVISION:-main}"
URL="http://localhost:8080/${MODEL}/resolve/${REVISION}/${FILE}"

kubectl="kubectl --context ${CTX} -n ${NS}"

echo "=== CoreWeave model-mesh e2e test ==="
echo "  model:    ${MODEL}"
echo "  file:     ${FILE}"
echo "  revision: ${REVISION}"
echo ""

# Check pods are running
echo "--- Pod status ---"
${kubectl} get pods -o wide
echo ""

running=$(${kubectl} get pods --field-selector=status.phase=Running -o name | wc -l | tr -d ' ')
if [ "${running}" -lt 2 ]; then
    echo "ERROR: need at least 2 running pods, got ${running}"
    exit 1
fi

# Check transport availability
echo "--- Transport status ---"
for pod in model-mesh-0 model-mesh-1; do
    echo "${pod}:"
    ${kubectl} logs "${pod}" 2>&1 | grep -E "transport (available|unavailable|probe complete)" | sed 's/^/  /'
    echo ""
done

# Step 1: Seed on pod-0 (pulls from HF upstream)
echo "--- Step 1: Seed on pod-0 (HF upstream) ---"
${kubectl} exec model-mesh-0 -- \
    curl -sS -w '\n  status: %{http_code}\n  time: %{time_total}s\n  size: %{size_download} bytes\n' \
    -o /dev/null "${URL}"
echo ""

# Wait for gossip to propagate the content advertisement
echo "Waiting 5s for gossip propagation..."
sleep 5

# Step 2: Fetch on pod-1 (should discover pod-0 via gossip)
echo "--- Step 2: Fetch on pod-1 (peer mesh) ---"
${kubectl} exec model-mesh-1 -- \
    curl -sS -w '\n  status: %{http_code}\n  time: %{time_total}s\n  size: %{size_download} bytes\n' \
    -o /dev/null "${URL}"
echo ""

# Check which transport was used on pod-1
echo "--- Pod-1 fetch details ---"
${kubectl} logs model-mesh-1 2>&1 | grep -E "fetched|fetch.*peer|peer fetch|transport|RDMA|NIXL|nixl" | tail -10 | sed 's/^/  /'
echo ""

# Step 3: Second fetch on pod-1 (should be local cache hit)
echo "--- Step 3: Re-fetch on pod-1 (local cache) ---"
${kubectl} exec model-mesh-1 -- \
    curl -sS -w '\n  status: %{http_code}\n  time: %{time_total}s\n  size: %{size_download} bytes\n' \
    -o /dev/null "${URL}"
echo ""

echo "=== Test complete ==="
