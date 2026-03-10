#!/usr/bin/env bash
# End-to-end test for NIXL GPUDirect P2P weight transfer on CoreWeave.
#
# Architecture: each pod has two containers:
#   - model-mesh (sidecar): daemon for HF proxy, gossip, NIXL metadata, IPC
#   - vllm: serves inference with --load-format=layercast plugin
#
# Flow:
#   1. Deploy vllm StatefulSet (2 pods, each with model-mesh sidecar)
#   2. vllm-0 loads Qwen2.5-3B from HF via model-mesh proxy, seeds VRAM
#   3. vllm-0 publishes NIXL metadata + weight manifest via daemon gossip
#   4. vllm-1 discovers peer metadata, loads weights via NIXL GPUDirect from vllm-0
#   5. Both pods serve inference, we verify with a completion request
#
# Usage:
#   ./scripts/test-nixl-e2e.sh              # full deploy + test
#   ./scripts/test-nixl-e2e.sh --no-deploy  # skip deploy, just run tests
#   ./scripts/test-nixl-e2e.sh --cleanup    # tear everything down

set -euo pipefail

# Override with KUBE_CONTEXT env var to target a different cluster.
CTX="${KUBE_CONTEXT:-coreweave-waldorf}"
NS="layercast"
MODEL="Qwen/Qwen2.5-3B"
DEPLOY_DIR="deploy/nixl-e2e"
DEPLOY="${1:-}"

kubectl="kubectl --context ${CTX} -n ${NS}"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
fail()  { echo "FAIL: $*" >&2; exit 1; }

wait_for_pod() {
    local pod="$1" timeout="$2"
    info "Waiting for ${pod} to be Ready (timeout ${timeout}s)..."
    local elapsed=0
    while [ "${elapsed}" -lt "${timeout}" ]; do
        local ready
        ready=$(${kubectl} get pod "${pod}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
        if [ "${ready}" = "True" ]; then
            info "${pod} is ready."
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        # Show container statuses
        local statuses
        statuses=$(${kubectl} get pod "${pod}" -o jsonpath='{range .status.containerStatuses[*]}{.name}={.ready} {end}' 2>/dev/null || echo "pending")
        echo "  ... ${statuses} (${elapsed}s elapsed)"
    done
    warn "Timed out waiting for ${pod}."
    ${kubectl} describe pod "${pod}" | tail -30
    return 1
}

# --------------------------------------------------------------------------
# Cleanup mode
# --------------------------------------------------------------------------

if [ "${DEPLOY}" = "--cleanup" ]; then
    info "Tearing down NIXL e2e test..."
    kubectl --context "${CTX}" delete -k "${DEPLOY_DIR}" --ignore-not-found
    info "Cleanup complete."
    exit 0
fi

# --------------------------------------------------------------------------
# Deploy
# --------------------------------------------------------------------------

if [ "${DEPLOY}" != "--no-deploy" ]; then
    info "Deploying NIXL e2e stack..."
    kubectl --context "${CTX}" apply -k "${DEPLOY_DIR}"
    echo ""
fi

# --------------------------------------------------------------------------
# Wait for vllm-0 (seeder: loads from HF, publishes NIXL metadata)
# --------------------------------------------------------------------------

info "Waiting for vllm-0 to fully start (model-mesh sidecar + vLLM)..."
wait_for_pod "vllm-0" 600 || {
    warn "vllm-0 not ready. Dumping logs..."
    echo "--- model-mesh sidecar ---"
    ${kubectl} logs vllm-0 -c model-mesh --tail=20 2>&1 | sed 's/^/    /'
    echo "--- vllm ---"
    ${kubectl} logs vllm-0 -c vllm --tail=30 2>&1 | sed 's/^/    /'
    fail "vllm-0 did not become ready in time"
}
echo ""

# Verify vllm-0 serves inference
info "Verifying vllm-0 serves inference..."
${kubectl} exec vllm-0 -c vllm -- curl -sS -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 5}" \
    | python3 -m json.tool
echo ""

# --------------------------------------------------------------------------
# Check NIXL metadata on vllm-0
# --------------------------------------------------------------------------

info "Checking vllm-0 NIXL/layercast logs..."
echo "  --- vllm container ---"
${kubectl} logs vllm-0 -c vllm 2>&1 | grep -iE "nixl|publish|vram|metadata|layercast" | tail -10 | sed 's/^/    /'
echo "  --- model-mesh sidecar ---"
${kubectl} logs vllm-0 -c model-mesh 2>&1 | grep -iE "nixl|vram|publish" | tail -5 | sed 's/^/    /'
echo ""

# --------------------------------------------------------------------------
# Wait for vllm-1 (should load via NIXL GPUDirect P2P from vllm-0)
# --------------------------------------------------------------------------

info "Waiting for vllm-1 (expecting NIXL GPUDirect P2P load)..."
wait_for_pod "vllm-1" 600 || {
    warn "vllm-1 not ready. Dumping logs..."
    echo "--- model-mesh sidecar ---"
    ${kubectl} logs vllm-1 -c model-mesh --tail=20 2>&1 | sed 's/^/    /'
    echo "--- vllm ---"
    ${kubectl} logs vllm-1 -c vllm --tail=50 2>&1 | sed 's/^/    /'
    fail "vllm-1 did not become ready in time"
}
echo ""

# --------------------------------------------------------------------------
# Verify NIXL transfer on vllm-1
# --------------------------------------------------------------------------

info "Checking vllm-1 load method..."
${kubectl} logs vllm-1 -c vllm 2>&1 | grep -iE "nixl|gpudirect|p2p|layercast|transfer|fallback|daemon" | tail -15 | sed 's/^/    /'
echo ""

info "Verifying vllm-1 serves inference..."
${kubectl} exec vllm-1 -c vllm -- curl -sS -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 5}" \
    | python3 -m json.tool
echo ""

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

echo ""
info "=== NIXL GPUDirect P2P E2E Test Complete ==="
echo ""
echo "  Model:  ${MODEL}"
echo "  vllm-0: Loaded from HF (seeder)"
echo "  vllm-1: Should have loaded via NIXL GPUDirect P2P"
echo ""
echo "  Quick log commands:"
echo "    ${kubectl} logs vllm-0 -c vllm | grep -i nixl"
echo "    ${kubectl} logs vllm-1 -c vllm | grep -i nixl"
echo "    ${kubectl} logs vllm-0 -c model-mesh | grep -i nixl"
echo ""
