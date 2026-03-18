#!/usr/bin/env bash
# End-to-end test for NIXL GPUDirect P2P weight transfer on CoreWeave.
#
# Architecture:
#   - Central metadata-server: leader-elected, manages peer registry + NIXL metadata
#   - vllm pods: single container, talks to metadata server via gRPC
#
# Flow:
#   1. Deploy metadata-server + vllm seed/consumer
#   2. vllm-seed loads from HF, publishes NIXL VRAM metadata via gRPC
#   3. vllm-consumer discovers seed via metadata server, loads via NIXL GPUDirect
#   4. Both pods serve inference, we verify with a completion request
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

wait_for_deployment() {
    local name="$1" timeout="$2"
    info "Waiting for ${name} to be Ready (timeout ${timeout}s)..."
    local elapsed=0
    while [ "${elapsed}" -lt "${timeout}" ]; do
        local pod
        pod=$(${kubectl} get pods -l "app=vllm,app.kubernetes.io/component=${name}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        if [ -n "${pod}" ]; then
            local ready
            ready=$(${kubectl} get pod "${pod}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
            if [ "${ready}" = "True" ]; then
                info "${name} (${pod}) is ready."
                return 0
            fi
            local statuses
            statuses=$(${kubectl} get pod "${pod}" -o jsonpath='{range .status.containerStatuses[*]}{.name}={.ready} {end}' 2>/dev/null || echo "pending")
            echo "  ... ${statuses} (${elapsed}s elapsed)"
        else
            echo "  ... waiting for pod (${elapsed}s elapsed)"
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    warn "Timed out waiting for ${name}."
    if [ -n "${pod:-}" ]; then
        ${kubectl} describe pod "${pod}" | tail -30
    fi
    return 1
}

get_pod() {
    local component="$1"
    ${kubectl} get pods -l "app=vllm,app.kubernetes.io/component=${component}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
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
# Wait for seed (loads from HF, publishes NIXL metadata)
# --------------------------------------------------------------------------

info "Waiting for vllm-seed to fully start..."
wait_for_deployment "seed" 600 || {
    pod=$(get_pod "seed")
    warn "seed not ready. Dumping logs..."
    ${kubectl} logs "${pod}" --tail=30 2>&1 | sed 's/^/    /'
    fail "seed did not become ready in time"
}
echo ""

# Verify seed serves inference
seed_pod=$(get_pod "seed")
info "Verifying seed serves inference..."
${kubectl} exec "${seed_pod}" -- curl -sS -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 5}" \
    | python3 -m json.tool
echo ""

# --------------------------------------------------------------------------
# Check NIXL metadata on seed
# --------------------------------------------------------------------------

info "Checking seed NIXL/layercast logs..."
${kubectl} logs "${seed_pod}" 2>&1 | grep -iE "nixl|publish|vram|metadata|layercast" | tail -10 | sed 's/^/    /'
echo ""

# --------------------------------------------------------------------------
# Wait for consumer (should load via NIXL GPUDirect P2P from seed)
# --------------------------------------------------------------------------

info "Waiting for vllm-consumer (expecting NIXL GPUDirect P2P load)..."
wait_for_deployment "consumer" 600 || {
    pod=$(get_pod "consumer")
    warn "consumer not ready. Dumping logs..."
    ${kubectl} logs "${pod}" --tail=50 2>&1 | sed 's/^/    /'
    fail "consumer did not become ready in time"
}
echo ""

# --------------------------------------------------------------------------
# Verify NIXL transfer on consumer
# --------------------------------------------------------------------------

consumer_pod=$(get_pod "consumer")
info "Checking consumer load method..."
${kubectl} logs "${consumer_pod}" 2>&1 | grep -iE "nixl|gpudirect|p2p|layercast|transfer|fallback" | tail -15 | sed 's/^/    /'
echo ""

info "Verifying consumer serves inference..."
${kubectl} exec "${consumer_pod}" -- curl -sS -X POST http://localhost:8000/v1/completions \
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
echo "  Model:    ${MODEL}"
echo "  Seed:     Loaded from HF (seeder)"
echo "  Consumer: Should have loaded via NIXL GPUDirect P2P"
echo ""
echo "  Quick log commands:"
echo "    ${kubectl} logs ${seed_pod} | grep -i nixl"
echo "    ${kubectl} logs ${consumer_pod} | grep -i nixl"
echo ""
