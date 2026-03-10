#!/usr/bin/env bash
# Model load benchmark: HF cold vs NFS PVC vs NIXL GPUDirect (Rust vs Python).
#
# Deploys each scenario on CoreWeave, measures time-to-ready, scrapes
# structured logs (captured via stern) from the layercast plugin, and
# outputs a markdown table.
#
# Usage:
#   ./scripts/benchmark.sh                          # run all scenarios
#   ./scripts/benchmark.sh --skip-hf                # skip HF cold (slow)
#   ./scripts/benchmark.sh --skip-nfs               # skip NFS
#   ./scripts/benchmark.sh --skip-nixl              # skip NIXL (Rust sidecar)
#   ./scripts/benchmark.sh --skip-nixl-scaling      # skip NIXL 3-replica scaling
#   ./scripts/benchmark.sh --skip-populate           # assume model already on PVC
#   ./scripts/benchmark.sh --model Qwen/Qwen2.5-3B  # use a smaller model
#   ./scripts/benchmark.sh --cleanup                 # tear everything down

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env or flags)
# ---------------------------------------------------------------------------

# Override with KUBE_CONTEXT env var to target a different cluster.
CTX="${KUBE_CONTEXT:-coreweave-waldorf}"
NS="${BENCH_NS:-layercast}"

export BENCH_MODEL="${BENCH_MODEL:-Qwen/Qwen2.5-32B}"
export BENCH_MODEL_PATH="${BENCH_MODEL}"  # "/" is fine in paths
export VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/wseaton/layercast:vllm-plugin-main}"
export MESH_IMAGE="${MESH_IMAGE:-ghcr.io/wseaton/layercast:model-mesh-main}"

SKIP_HF=false
SKIP_NFS=false
SKIP_NIXL=false
SKIP_NIXL_SCALING=false
SKIP_POPULATE=false
CLEANUP_ONLY=false

DEPLOY_DIR="deploy/benchmark"
NIXL_E2E_DIR="deploy/nixl-e2e"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-hf)       SKIP_HF=true ;;
        --skip-nfs)      SKIP_NFS=true ;;
        --skip-nixl)     SKIP_NIXL=true ;;
        --skip-nixl-scaling) SKIP_NIXL_SCALING=true ;;
        --skip-populate) SKIP_POPULATE=true ;;
        --cleanup)       CLEANUP_ONLY=true ;;
        --model)         shift; export BENCH_MODEL="$1"; export BENCH_MODEL_PATH="$1" ;;
        --vllm-image)    shift; export VLLM_IMAGE="$1" ;;
        --mesh-image)    shift; export MESH_IMAGE="$1" ;;
        --context)       shift; CTX="$1" ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

# Kustomize image overrides (derived from VLLM_IMAGE and MESH_IMAGE after flag parsing).
# Splits "registry/repo:tag" into newName and newTag for the kustomize images transformer.
VLLM_IMAGE_NAME="${VLLM_IMAGE%:*}"
VLLM_IMAGE_TAG="${VLLM_IMAGE##*:}"
MESH_IMAGE_NAME="${MESH_IMAGE%:*}"
MESH_IMAGE_TAG="${MESH_IMAGE##*:}"

kubectl="kubectl --context ${CTX} -n ${NS}"

# ---------------------------------------------------------------------------
# Log directory (stern captures go here, survive pod deletion)
# ---------------------------------------------------------------------------

LOG_DIR="benchmark-logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOG_DIR}"

# Track stern PIDs so we can clean them up
declare -a STERN_PIDS=()

# Start stern to capture all pod logs for a scenario.
# Usage: start_stern <scenario_label> <log_file_name>
start_stern() {
    local label="$1" logfile="${LOG_DIR}/$2"
    stern --context "${CTX}" -n "${NS}" -l "bench/scenario=${label}" \
        --output raw --no-follow=false \
        > "${logfile}" 2>/dev/null &
    local pid=$!
    STERN_PIDS+=("${pid}")
    info "stern capturing logs for ${label} (pid ${pid}) -> ${logfile}"
}

# Stop all running stern processes.
stop_stern() {
    if [[ ${#STERN_PIDS[@]} -gt 0 ]]; then
        for pid in "${STERN_PIDS[@]}"; do
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        done
    fi
    STERN_PIDS=()
}

# Cleanup stern on exit
trap 'stop_stern' EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "==> $*" >&2; }
warn()  { echo "WARNING: $*" >&2; }
fail()  { echo "FAIL: $*" >&2; exit 1; }

# Only substitute model variables, leave shell vars like ${TARGET} alone.
# Images are no longer envsubst vars; they're set via kustomize (see below).
ENVSUBST_VARS='${BENCH_MODEL} ${BENCH_MODEL_PATH}'

# Apply a YAML file: envsubst for model vars, then kustomize for image overrides.
# Creates a temporary kustomization overlay so kustomize can rewrite the generic
# image names (model-mesh:latest, vllm-plugin:latest) to real registry/tag values.
apply_template() {
    local file="$1"
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf ${tmpdir}" RETURN

    # Expand model variables into the temp dir
    envsubst "${ENVSUBST_VARS}" < "${file}" > "${tmpdir}/resource.yaml"

    # Write a kustomization that sets images
    cat > "${tmpdir}/kustomization.yaml" <<KEOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - resource.yaml
images:
  - name: model-mesh
    newName: ${MESH_IMAGE_NAME}
    newTag: "${MESH_IMAGE_TAG}"
  - name: vllm-plugin
    newName: ${VLLM_IMAGE_NAME}
    newTag: "${VLLM_IMAGE_TAG}"
KEOF

    kubectl --context "${CTX}" -n "${NS}" apply -k "${tmpdir}"
}

delete_template() {
    local file="$1"
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf ${tmpdir}" RETURN

    envsubst "${ENVSUBST_VARS}" < "${file}" > "${tmpdir}/resource.yaml"

    cat > "${tmpdir}/kustomization.yaml" <<KEOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - resource.yaml
images:
  - name: model-mesh
    newName: ${MESH_IMAGE_NAME}
    newTag: "${MESH_IMAGE_TAG}"
  - name: vllm-plugin
    newName: ${VLLM_IMAGE_NAME}
    newTag: "${VLLM_IMAGE_TAG}"
KEOF

    kubectl --context "${CTX}" -n "${NS}" delete --ignore-not-found -k "${tmpdir}"
}

# Get pod name for a bench scenario label.
get_pod() {
    local scenario="$1"
    ${kubectl} get pods -l "bench/scenario=${scenario}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

# Wait for a deployment's pod to become Ready.
# Prints elapsed seconds to stdout on success, -1 on timeout.
# All progress output goes to stderr so $() captures only the number.
wait_for_ready() {
    local scenario="$1"
    local timeout="${2:-900}"
    local start_epoch
    start_epoch=$(date +%s)

    info "Waiting for bench/${scenario} to be Ready (timeout ${timeout}s)..."

    local elapsed=0
    while [ "${elapsed}" -lt "${timeout}" ]; do
        local pod
        pod=$(get_pod "${scenario}")
        if [ -n "${pod}" ]; then
            local ready
            ready=$(${kubectl} get pod "${pod}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
            if [ "${ready}" = "True" ]; then
                local end_epoch
                end_epoch=$(date +%s)
                local total=$((end_epoch - start_epoch))
                info "${scenario} ready in ${total}s"
                echo "${total}"
                return 0
            fi
            # Show progress on stderr
            local statuses
            statuses=$(${kubectl} get pod "${pod}" -o jsonpath='{range .status.containerStatuses[*]}{.name}={.ready} {end}' 2>/dev/null || echo "pending")
            echo "  ... ${statuses} (${elapsed}s)" >&2
        else
            echo "  ... waiting for pod (${elapsed}s)" >&2
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done

    warn "Timed out waiting for ${scenario}"
    local pod
    pod=$(get_pod "${scenario}")
    if [ -n "${pod}" ]; then
        ${kubectl} describe pod "${pod}" | tail -20 >&2
    fi
    echo "-1"
    return 1
}

# Extract structured log fields from a log file (captured by stern).
# Usage: extract_log_field_from_file <log_file> <event_name> <field_name>
extract_log_field_from_file() {
    local logfile="$1" event="$2" field="$3"
    if [ ! -f "${logfile}" ]; then
        echo "N/A"
        return
    fi
    local result
    # Stern raw output: lines may have vLLM process prefixes before JSON.
    # NO_COLOR=1 on vllm containers removes ANSI codes.
    # Find line with the event, extract the JSON object, read the field.
    result=$(grep "\"event\": \"${event}\"" "${logfile}" \
        | tail -1 \
        | python3 -c "
import sys, json, re
line = re.sub(r'\x1b\[[0-9;]*m', '', sys.stdin.readline())
m = re.search(r'\{.*\}', line)
if m:
    d = json.loads(m.group())
    v = d.get('${field}')
    if v is not None: print(v)
" 2>/dev/null) || true
    echo "${result:-N/A}"
}

# Extract the creation-to-ready duration from K8s timestamps (more precise).
get_k8s_ready_duration() {
    local pod="$1"
    local created ready_time
    created=$(${kubectl} get pod "${pod}" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "")
    ready_time=$(${kubectl} get pod "${pod}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}' 2>/dev/null || echo "")
    if [ -n "${created}" ] && [ -n "${ready_time}" ]; then
        python3 -c "
from datetime import datetime
c = datetime.fromisoformat('${created}'.replace('Z','+00:00'))
r = datetime.fromisoformat('${ready_time}'.replace('Z','+00:00'))
print(f'{(r-c).total_seconds():.1f}')
" 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

# ---------------------------------------------------------------------------
# Cleanup mode
# ---------------------------------------------------------------------------

if [ "${CLEANUP_ONLY}" = true ]; then
    info "Cleaning up all benchmark resources..."
    delete_template "${DEPLOY_DIR}/nixl-p2p.yaml" 2>/dev/null || true
    delete_template "${DEPLOY_DIR}/nixl-scaling.yaml" 2>/dev/null || true
    delete_template "${DEPLOY_DIR}/nfs-cached.yaml" 2>/dev/null || true
    delete_template "${DEPLOY_DIR}/hf-cold.yaml" 2>/dev/null || true
    ${kubectl} delete job populate-model --ignore-not-found 2>/dev/null || true
    ${kubectl} delete -f "${DEPLOY_DIR}/pvc.yaml" --ignore-not-found 2>/dev/null || true
    info "Cleanup complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

info "Benchmark configuration:"
echo "  Model:      ${BENCH_MODEL}" >&2
echo "  vLLM image: ${VLLM_IMAGE}" >&2
echo "  Mesh image: ${MESH_IMAGE}" >&2
echo "  Context:    ${CTX}" >&2
echo "  Namespace:  ${NS}" >&2
echo "  Log dir:    ${LOG_DIR}" >&2
echo "" >&2

# Ensure namespace exists
kubectl --context "${CTX}" get ns "${NS}" >/dev/null 2>&1 || \
    kubectl --context "${CTX}" create ns "${NS}"

# Results arrays
declare -a RESULT_SCENARIO=()
declare -a RESULT_TTR=()
declare -a RESULT_WEIGHT_TIME=()
declare -a RESULT_TRANSFER_GBPS=()
declare -a RESULT_CHECKSUM=()
declare -a RESULT_METHOD=()

# ---------------------------------------------------------------------------
# Populate NFS PVC (shared between NFS and NIXL scenarios)
# ---------------------------------------------------------------------------

if [ "${SKIP_POPULATE}" = false ] && { [ "${SKIP_NFS}" = false ] || [ "${SKIP_NIXL}" = false ]; }; then
    info "Ensuring PVC exists..."
    ${kubectl} apply -f "${DEPLOY_DIR}/pvc.yaml"

    info "Populating NFS PVC with ${BENCH_MODEL}..."
    ${kubectl} delete job populate-model --ignore-not-found 2>/dev/null || true
    apply_template "${DEPLOY_DIR}/populate-job.yaml"

    info "Waiting for populate job to complete (this may take a while for large models)..."
    if ! ${kubectl} wait --for=condition=complete job/populate-model --timeout=3600s 2>/dev/null; then
        warn "Populate job may still be running or failed"
        ${kubectl} logs job/populate-model --tail=20 2>/dev/null || true
    fi
    echo "" >&2
fi

# ---------------------------------------------------------------------------
# Scenario 1: HF Cold
# ---------------------------------------------------------------------------

if [ "${SKIP_HF}" = false ]; then
    info "=== Scenario: HF Cold Download ==="

    delete_template "${DEPLOY_DIR}/hf-cold.yaml" 2>/dev/null || true
    sleep 2
    apply_template "${DEPLOY_DIR}/hf-cold.yaml"
    start_stern "hf-cold" "hf-cold.log"

    hf_ttr=$(wait_for_ready "hf-cold" 3600) || true

    hf_pod=$(get_pod "hf-cold")
    hf_k8s_ttr="N/A"
    if [ -n "${hf_pod}" ]; then
        hf_k8s_ttr=$(get_k8s_ready_duration "${hf_pod}")
    fi

    RESULT_SCENARIO+=("HF Cold")
    RESULT_TTR+=("${hf_k8s_ttr}")
    RESULT_WEIGHT_TIME+=("N/A")
    RESULT_TRANSFER_GBPS+=("N/A")
    RESULT_CHECKSUM+=("N/A")
    RESULT_METHOD+=("hf-download")

    stop_stern
    info "Cleaning up HF cold..."
    delete_template "${DEPLOY_DIR}/hf-cold.yaml" 2>/dev/null || true
    echo "" >&2
fi

# ---------------------------------------------------------------------------
# Scenario 2: NFS Cached
# ---------------------------------------------------------------------------

if [ "${SKIP_NFS}" = false ]; then
    info "=== Scenario: NFS Cached (VAST) ==="

    # Ensure PVC is there
    ${kubectl} apply -f "${DEPLOY_DIR}/pvc.yaml" 2>/dev/null || true

    ${kubectl} delete deployment bench-nfs-cached --ignore-not-found 2>/dev/null || true
    sleep 2
    apply_template "${DEPLOY_DIR}/nfs-cached.yaml"
    start_stern "nfs-cached" "nfs-cached.log"

    nfs_ttr=$(wait_for_ready "nfs-cached" 900) || true

    nfs_pod=$(get_pod "nfs-cached")
    nfs_k8s_ttr="N/A"
    if [ -n "${nfs_pod}" ]; then
        nfs_k8s_ttr=$(get_k8s_ready_duration "${nfs_pod}")
    fi

    RESULT_SCENARIO+=("NFS Cached (VAST)")
    RESULT_TTR+=("${nfs_k8s_ttr}")
    RESULT_WEIGHT_TIME+=("N/A")
    RESULT_TRANSFER_GBPS+=("N/A")
    RESULT_CHECKSUM+=("N/A")
    RESULT_METHOD+=("nfs-read")

    stop_stern
    info "Cleaning up NFS cached..."
    # Delete deployment only, keep PVC for NIXL seed
    ${kubectl} delete deployment bench-nfs-cached --ignore-not-found 2>/dev/null || true
    echo "" >&2
fi

# ---------------------------------------------------------------------------
# Scenario 3: NIXL GPUDirect P2P
# ---------------------------------------------------------------------------

if [ "${SKIP_NIXL}" = false ]; then
    info "=== Scenario: NIXL GPUDirect P2P ==="

    # Ensure CRD and RBAC are deployed (reuse from nixl-e2e)
    kubectl --context "${CTX}" apply -f "${NIXL_E2E_DIR}/crd.yaml" 2>/dev/null || true
    kubectl --context "${CTX}" apply -f "${NIXL_E2E_DIR}/rbac.yaml" -n "${NS}" 2>/dev/null || true

    # Ensure PVC exists
    ${kubectl} apply -f "${DEPLOY_DIR}/pvc.yaml" 2>/dev/null || true

    delete_template "${DEPLOY_DIR}/nixl-p2p.yaml" 2>/dev/null || true
    sleep 2
    apply_template "${DEPLOY_DIR}/nixl-p2p.yaml"

    # Capture logs for both seed and consumer
    start_stern "nixl-seed" "nixl-seed.log"
    start_stern "nixl-consumer" "nixl-consumer.log"

    info "Waiting for NIXL seed to be ready..."
    seed_ttr=$(wait_for_ready "nixl-seed" 900) || true
    echo "" >&2

    info "Waiting for NIXL consumer (this is the metric)..."
    nixl_ttr=$(wait_for_ready "nixl-consumer" 600) || true

    nixl_pod=$(get_pod "nixl-consumer")
    nixl_k8s_ttr="N/A"
    nixl_weight_time="N/A"
    nixl_gbps="N/A"
    nixl_method="nixl-gpudirect"

    if [ -n "${nixl_pod}" ]; then
        nixl_k8s_ttr=$(get_k8s_ready_duration "${nixl_pod}")
    fi

    # Give stern a moment to flush remaining logs
    sleep 3

    nixl_weight_time=$(extract_log_field_from_file "${LOG_DIR}/nixl-consumer.log" "model_loaded" "elapsed_s")
    nixl_gbps=$(extract_log_field_from_file "${LOG_DIR}/nixl-consumer.log" "nixl_transfer_complete" "gbps")
    nixl_checksum=$(extract_log_field_from_file "${LOG_DIR}/nixl-consumer.log" "checksum_verification" "elapsed_s")

    RESULT_SCENARIO+=("NIXL GPUDirect P2P")
    RESULT_TTR+=("${nixl_k8s_ttr}")
    RESULT_WEIGHT_TIME+=("${nixl_weight_time}")
    RESULT_TRANSFER_GBPS+=("${nixl_gbps}")
    RESULT_CHECKSUM+=("${nixl_checksum}")
    RESULT_METHOD+=("${nixl_method}")

    stop_stern
    info "Cleaning up NIXL benchmark..."
    delete_template "${DEPLOY_DIR}/nixl-p2p.yaml" 2>/dev/null || true
    echo "" >&2
fi

# ---------------------------------------------------------------------------
# Scenario 4: NIXL Scaling (seed + 2 consumers, sequential)
# ---------------------------------------------------------------------------

if [ "${SKIP_NIXL_SCALING}" = false ]; then
    info "=== Scenario: NIXL Scaling (seed -> c1 -> c2) ==="

    # Ensure CRD and RBAC are deployed
    kubectl --context "${CTX}" apply -f "${NIXL_E2E_DIR}/crd.yaml" 2>/dev/null || true
    kubectl --context "${CTX}" apply -f "${NIXL_E2E_DIR}/rbac.yaml" -n "${NS}" 2>/dev/null || true

    # Ensure PVC exists
    ${kubectl} apply -f "${DEPLOY_DIR}/pvc.yaml" 2>/dev/null || true

    delete_template "${DEPLOY_DIR}/nixl-scaling.yaml" 2>/dev/null || true
    sleep 2
    apply_template "${DEPLOY_DIR}/nixl-scaling.yaml"

    # Capture logs for all three pods
    start_stern "scale-seed" "scale-seed.log"
    start_stern "scale-c1" "scale-c1.log"
    start_stern "scale-c2" "scale-c2.log"

    # Wait for each in sequence (init containers enforce ordering)
    info "Waiting for scaling seed to be ready..."
    scale_seed_ttr=$(wait_for_ready "scale-seed" 900) || true
    echo "" >&2

    info "Waiting for scaling consumer-1 (NIXL + compile cache populate)..."
    scale_c1_ttr=$(wait_for_ready "scale-c1" 600) || true
    echo "" >&2

    info "Waiting for scaling consumer-2 (NIXL + compile cache hits)..."
    scale_c2_ttr=$(wait_for_ready "scale-c2" 600) || true

    # Give stern a moment to flush
    sleep 3

    # Scrape compile cache stats from each pod's model-mesh sidecar
    scrape_compile_stats() {
        local scenario="$1"
        local pod
        pod=$(get_pod "${scenario}")
        if [ -n "${pod}" ]; then
            ${kubectl} exec "${pod}" -c model-mesh -- \
                curl -sf http://127.0.0.1:8081/internal/compile-cache-stats 2>/dev/null || echo "{}"
        else
            echo "{}"
        fi
    }

    # Collect results for each pod
    for role in seed c1 c2; do
        local_scenario="scale-${role}"
        local_pod=$(get_pod "${local_scenario}")
        local_k8s_ttr="N/A"
        local_weight_time="N/A"
        local_gbps="N/A"
        local_checksum="N/A"

        if [ -n "${local_pod}" ]; then
            local_k8s_ttr=$(get_k8s_ready_duration "${local_pod}")
        fi

        local_weight_time=$(extract_log_field_from_file "${LOG_DIR}/${local_scenario}.log" "model_loaded" "elapsed_s")
        local_gbps=$(extract_log_field_from_file "${LOG_DIR}/${local_scenario}.log" "nixl_transfer_complete" "gbps")
        local_checksum=$(extract_log_field_from_file "${LOG_DIR}/${local_scenario}.log" "checksum_verification" "elapsed_s")

        # Compile cache stats
        local_cc_stats=$(scrape_compile_stats "${local_scenario}")
        local_cc_gets=$(echo "${local_cc_stats}" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('gets',0))" 2>/dev/null || echo "0")
        local_cc_hits=$(echo "${local_cc_stats}" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('mem_hits',0)+d.get('disk_hits',0)+d.get('peer_hits',0))" 2>/dev/null || echo "0")

        local_method="nixl-gpudirect"
        if [ "${role}" = "seed" ]; then
            local_method="hf-download"
        fi

        local_display="Scale ${role}"
        RESULT_SCENARIO+=("${local_display}")
        RESULT_TTR+=("${local_k8s_ttr}")
        RESULT_WEIGHT_TIME+=("${local_weight_time}")
        RESULT_TRANSFER_GBPS+=("${local_gbps}")
        RESULT_CHECKSUM+=("${local_checksum}")
        RESULT_METHOD+=("${local_method}")

        info "${local_display}: TTR=${local_k8s_ttr}s, compile cache ${local_cc_hits}/${local_cc_gets} hits"
    done

    stop_stern
    info "Cleaning up NIXL scaling benchmark..."
    delete_template "${DEPLOY_DIR}/nixl-scaling.yaml" 2>/dev/null || true
    echo "" >&2
fi

# ---------------------------------------------------------------------------
# Results (to stdout so they can be piped/captured)
# ---------------------------------------------------------------------------

echo ""
echo "## Benchmark Results"
echo ""
echo "**Model**: \`${BENCH_MODEL}\`"
echo "**Cluster**: ${CTX}"
echo "**Date**: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "**Logs**: \`${LOG_DIR}/\`"
echo ""

# Markdown table
printf "| %-25s | %-15s | %-18s | %-12s | %-14s | %-15s |\n" \
    "Scenario" "Time-to-Ready" "Weight Load (s)" "Throughput" "Checksum (s)" "Method"
printf "| %-25s | %-15s | %-18s | %-12s | %-14s | %-15s |\n" \
    "-------------------------" "---------------" "------------------" "------------" "--------------" "---------------"

for i in "${!RESULT_SCENARIO[@]}"; do
    ttr="${RESULT_TTR[$i]}"
    if [ "${ttr}" != "N/A" ] && [ "${ttr}" != "-1" ]; then
        ttr="${ttr}s"
    fi
    wt="${RESULT_WEIGHT_TIME[$i]}"
    if [ "${wt}" != "N/A" ]; then
        wt="${wt}s"
    fi
    gbps="${RESULT_TRANSFER_GBPS[$i]}"
    if [ "${gbps}" != "N/A" ]; then
        gbps="${gbps} GB/s"
    fi
    cksum="${RESULT_CHECKSUM[$i]}"
    if [ "${cksum}" != "N/A" ]; then
        cksum="${cksum}s"
    fi
    printf "| %-25s | %-15s | %-18s | %-12s | %-14s | %-15s |\n" \
        "${RESULT_SCENARIO[$i]}" "${ttr}" "${wt}" "${gbps}" "${cksum}" "${RESULT_METHOD[$i]}"
done

echo ""
info "Logs saved to ${LOG_DIR}/"
info "Done."
