#!/bin/bash
# =============================================================================
# Generic vLLM launcher driven by a deploy plan from aeon.core.deploy_planner.
#
#   dual     two nodes (TP=1, one per GPU) + adaptive_lb.py router   (Tier A)
#   split    one node, --tensor-parallel-size 2                      (Tier B)
#   offload  one node, TP=2 + --cpu-offload-gb                       (Tier C)
#
# MTP via --speculative-config when AEON_MTP_DRAFT_MODEL is set. vLLM fetches the
# weights from the HF hub at runtime (cached under ~/.cache/huggingface).
# Overrides (env): GPU_MEM_UTIL, MAX_MODEL_LEN.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${AEON_DEPLOY_PLAN:-}" ]; then
    echo "[adaptive-vllm] ERROR: AEON_DEPLOY_PLAN not set" >&2; exit 1
fi
pf() { python3 -c "import os,json;p=json.loads(os.environ['AEON_DEPLOY_PLAN']);print(p$1)"; }
TIER=$(pf "['tier']")
IMAGE=$(pf "['image']")
LB_PORT=$(pf "['lb_port']")
HF_MODEL="$AEON_HF_MODEL"
SERVED="$AEON_SERVED_NAME"

case "$TIER" in
    solo)    DEF_UTIL=0.85 ;;   # GPU0 to ourselves; leave room for MTP draft + activations
    dual)    DEF_UTIL=0.85 ;;
    split)   DEF_UTIL=0.90 ;;
    *)       DEF_UTIL=0.92 ;;
esac
# Precedence: explicit GPU_MEM_UTIL override > planner's fitted util (solo/dual,
# sized to weights+KV(ctx)+headroom instead of the whole card) > tier default.
UTIL="${GPU_MEM_UTIL:-${AEON_GPU_MEM_UTIL:-$DEF_UTIL}}"

SPEC_ARGS=()
if [ -n "${AEON_MTP_DRAFT_MODEL}" ]; then
    # SEPARATE draft/speculator model (e.g. the gemma-4 'assistant'). method is
    # plan-driven: gemma-4 assistant => "mtp" (native MTP speculator), other drafts =>
    # "draft_model". Passing "draft_model" for a gemma-4 assistant silently disables
    # MTP (vLLM #42005), so the catalog sets this explicitly.
    SPEC_ARGS=(--speculative-config "{\"method\": \"${AEON_MTP_METHOD:-draft_model}\", \"model\": \"${AEON_MTP_DRAFT_MODEL}\", \"num_speculative_tokens\": ${AEON_MTP_NMAX:-5}}")
elif { [ "${AEON_MTP_METHOD:-}" = "mtp" ] || [ "${AEON_MTP_METHOD:-}" = "eagle" ] || [ "${AEON_MTP_METHOD:-}" = "qwen3_5_mtp" ]; } && [ "${AEON_MTP_NMAX:-0}" != "0" ]; then
    # NATIVE in-checkpoint MTP (e.g. Qwen3.6): the MTP/EAGLE head ships inside the
    # target weights, so there is NO separate draft model -> omit "model" (passing one
    # would point vLLM at a nonexistent draft). vLLM auto-loads the head from the model.
    SPEC_ARGS=(--speculative-config "{\"method\": \"${AEON_MTP_METHOD}\", \"num_speculative_tokens\": ${AEON_MTP_NMAX}}")
fi

# Blackwell perf knobs (local, no re-quant):
#  - FP8 KV cache when the plan requests it (catalog kv_quant -> AEON_KV_QUANT, e.g. "fp8"):
#    ~halves KV memory so more context/batch fits and decode is faster at long ctx.
#  - FlashInfer attention backend: Blackwell-tuned FP8 attention kernels (NVFP4 only
#    accelerates the GEMMs; this puts attention on the FP8 units too). Override-able.
KV_ARGS=()
if [ -n "${AEON_KV_QUANT:-}" ]; then
    KV_ARGS=(--kv-cache-dtype "${AEON_KV_QUANT}")
fi
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# FlashInfer JIT-compiles the NVFP4 (sm_120 / Blackwell) FP4 GEMM kernels on first
# launch. Those CUTLASS templates are RAM-hungry to compile and FlashInfer fans out
# one ninja job per CPU by default -- on a 24-core box with no swap that runs ~24
# concurrent nvcc, each several GiB, and the compiler dies with "catastrophic error:
# out of memory", taking engine-core init down with it. Cap the concurrency (the
# compile is one-time and then cached). Override with AEON_JIT_MAX_JOBS if desired.
JIT_MAX_JOBS="${AEON_JIT_MAX_JOBS:-4}"

wait_for_health() {
    local name=$1 port=$2 count=0
    while true; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then return 1; fi
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${port}/health 2>/dev/null || true)" = "200" ]; then return 0; fi
        sleep 5; count=$((count+1))
        # First launch JIT-compiles FP4 kernels at capped parallelism (see JIT_MAX_JOBS),
        # which is slow but one-time/cached; allow ~35 min before giving up.
        [ $count -ge 420 ] && return 2
        [ $((count % 6)) -eq 0 ] && echo "[adaptive-vllm] ${name} still loading... ($((count*5))s)"
    done
}

launch_node() {
    local name=$1 devices=$2 port=$3 ctx=$4 tp=$5 cpu_offload=$6
    docker rm -f "$name" >/dev/null 2>&1 || true
    local args=(--model "$HF_MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$port"
                --tensor-parallel-size "$tp" --gpu-memory-utilization "$UTIL"
                --enable-prefix-caching --enable-chunked-prefill --max-model-len "${MAX_MODEL_LEN:-$ctx}")
    # Prefill batch size: with chunked prefill on, vLLM's small default (~2048) splits a
    # 20-30k-token agent prompt into many scheduler steps -> long time-to-first-token.
    # The catalog sets this high on roomy models so the prompt prefills in ~one pass.
    # Env MAX_NUM_BATCHED_TOKENS overrides; unset on both -> vLLM default (unchanged).
    local batched="${MAX_NUM_BATCHED_TOKENS:-${AEON_MAX_NUM_BATCHED:-}}"
    [ -n "$batched" ] && args+=(--max-num-batched-tokens "$batched")
    [ -n "$cpu_offload" ] && [ "$cpu_offload" != "0.0" ] && [ "$cpu_offload" != "0" ] && args+=(--cpu-offload-gb "$cpu_offload")
    args+=("${SPEC_ARGS[@]}")
    args+=("${KV_ARGS[@]}")
    echo "[adaptive-vllm] launch $name (GPU $devices, port $port, TP=$tp, util $UTIL, ctx $ctx${cpu_offload:+, cpu_offload ${cpu_offload}GiB})"
    # Persist ~/.cache/flashinfer too: otherwise the JIT-compiled FP4 kernels are lost
    # when the container is removed and every launch recompiles (slow + the OOM risk).
    mkdir -p "$HOME/.cache/flashinfer"
    docker run -d --label owner=aday --name "$name" --gpus "\"device=${devices}\"" --ipc=host -p "${port}:${port}" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -v "$HOME/.cache/triton:/root/.triton" -v "$HOME/.cache/vllm:/root/.cache/vllm" \
        -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
        -e TRITON_CACHE_DIR=/root/.triton -e VLLM_CACHE_ROOT=/root/.cache/vllm \
        -e VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
        -e MAX_JOBS="$JIT_MAX_JOBS" -e NVCC_THREADS=1 \
        "$IMAGE" "${args[@]}" >/dev/null 2>/tmp/aeon_${name}.err || {
            echo "[adaptive-vllm] docker run failed for $name:"; cat /tmp/aeon_${name}.err; return 1; }
}

NODE_LINES=$(python3 -c "
import os,json
p=json.loads(os.environ['AEON_DEPLOY_PLAN'])
for nd in p['nodes']:
    # tensor-parallel size = number of GPUs this node spans (solo/dual=1, split/offload=2)
    tp = len(str(nd['devices']).split(','))
    print('|'.join([nd['container'], str(nd['devices']), str(nd['port']), str(nd['ctx']), str(tp), str(nd.get('cpu_offload_gib',''))]))
")

NODE_URLS=()
while IFS='|' read -r name devices port ctx tp cpu_offload; do
    [ -z "$name" ] && continue
    launch_node "$name" "$devices" "$port" "$ctx" "$tp" "$cpu_offload" || exit 1
    NODE_URLS+=("http://127.0.0.1:${port}")
done <<< "$NODE_LINES"

echo "[adaptive-vllm] Waiting for vLLM node(s) to load (several minutes on first run)..."
while IFS='|' read -r name devices port ctx tp cpu_offload; do
    [ -z "$name" ] && continue
    wait_for_health "$name" "$port" || {
        crash_log="/tmp/aeon_${name}.crash.log"
        echo "[adaptive-vllm] ERROR: $name failed. Last 80 log lines (also saved to ${crash_log}):"
        # NOTE: --tail must precede the container name; trailing flags are rejected by
        # the docker CLI and were silently swallowed. tee to a host file so the reason
        # survives the session teardown that removes the container moments later.
        docker logs --tail 80 "$name" 2>&1 | tee "$crash_log"
        exit 1
    }
    echo "[adaptive-vllm] $name READY"
done <<< "$NODE_LINES"

if [ "$TIER" = "dual" ]; then
    LB_NAME=$(pf "['container_name']")
    docker rm -f "$LB_NAME" >/dev/null 2>&1 || true
    JOINED=$(IFS=,; echo "${NODE_URLS[*]}")
    echo "[adaptive-vllm] Starting router $LB_NAME on :$LB_PORT -> $JOINED"
    docker run -d --label owner=aday --name "$LB_NAME" --network host \
        -e AEON_LB_NODES="$JOINED" -e AEON_LB_PORT="$LB_PORT" \
        -v "${SCRIPT_DIR}/adaptive_lb.py:/app/adaptive_lb.py" -w /app \
        python:3.11-slim sh -c "pip install -q fastapi uvicorn httpx && python adaptive_lb.py" >/dev/null
    for i in $(seq 1 30); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${LB_PORT}/health 2>/dev/null || true)" = "200" ] && break
        sleep 2
    done
fi

echo "[adaptive-vllm] ONLINE: ${SERVED} (tier $TIER) on http://localhost:${LB_PORT}"
