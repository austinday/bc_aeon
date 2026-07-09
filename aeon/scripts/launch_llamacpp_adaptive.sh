#!/bin/bash
# =============================================================================
# Generic llama.cpp launcher driven by a deploy plan from aeon.core.deploy_planner.
#
# Reads AEON_DEPLOY_PLAN (JSON) + AEON_* env and deploys the model in one of:
#   dual     two copies (one per GPU) + adaptive_lb.py router  (Tier A)
#   split    one instance, GPU0-weighted --tensor-split        (Tier B)
#   offload  one instance, reduced -ngl (weights spill to RAM) (Tier C)
#
# MTP (--spec-type draft-mtp) is added only when AEON_MTP_DRAFT_FILE is set.
# Per-node OOM-backoff: if a container dies during load, ctx is halved (floor
# 64k) and the node is relaunched, so VRAM estimates self-correct on new cards.
#
# Tunable overrides (env): CTX, TENSOR_SPLIT, NGL, BATCH, UBATCH, KV_QUANT.
# =============================================================================
set -e

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIN_CTX=65536

if [ -z "${AEON_DEPLOY_PLAN:-}" ]; then
    echo "[adaptive-llamacpp] ERROR: AEON_DEPLOY_PLAN not set" >&2; exit 1
fi

pf() { python3 -c "import os,json;p=json.loads(os.environ['AEON_DEPLOY_PLAN']);print(p$1)"; }
TIER=$(pf "['tier']")
IMAGE=$(pf "['image']")
LB_PORT=$(pf "['lb_port']")
MTP=$(pf "['mtp']")

MODELS_DIR="$AEON_HOME/models/${AEON_MODEL_DIR}"
PHYSICAL_CORES=$(lscpu -b -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || nproc)
KV_QUANT="${KV_QUANT:-${AEON_KV_QUANT}}"

if [ ! -d "$MODELS_DIR" ]; then
    echo "[adaptive-llamacpp] ERROR: model dir not found: $MODELS_DIR" >&2; exit 1
fi
TARGET=$(cd "$MODELS_DIR" && find . -maxdepth 2 -name "$AEON_TARGET_GLOB" 2>/dev/null | head -1 | sed 's|^\./||')
if [ -z "$TARGET" ]; then
    echo "[adaptive-llamacpp] ERROR: no target GGUF matching '$AEON_TARGET_GLOB' in $MODELS_DIR" >&2; exit 1
fi
echo "[adaptive-llamacpp] tier=$TIER image=$IMAGE target=$TARGET kv_quant=${KV_QUANT:-default}"

# Vision (multimodal projector) + explicit chat template, when the catalog entry
# provides them. Both must live INSIDE model_dir (only that dir is mounted).
MMPROJ=""
if [ -n "${AEON_MMPROJ_FILE:-}" ]; then
    if [ -f "$MODELS_DIR/$AEON_MMPROJ_FILE" ]; then
        MMPROJ="$AEON_MMPROJ_FILE"
        echo "[adaptive-llamacpp] vision: mmproj=$MMPROJ"
    else
        echo "[adaptive-llamacpp] WARN: mmproj $AEON_MMPROJ_FILE missing; serving TEXT-ONLY" >&2
    fi
fi
CHAT_TEMPLATE=""
if [ -n "${AEON_CHAT_TEMPLATE_FILE:-}" ]; then
    if [ -f "$MODELS_DIR/$AEON_CHAT_TEMPLATE_FILE" ]; then
        CHAT_TEMPLATE="$AEON_CHAT_TEMPLATE_FILE"
    else
        echo "[adaptive-llamacpp] WARN: chat template $AEON_CHAT_TEMPLATE_FILE missing; using the GGUF's embedded one" >&2
    fi
fi

# --- MTP draft (self-heal: normalize the raw assistant GGUF if needed) ---
DRAFT=""
if [ "$MTP" = "True" ] && [ -n "${AEON_MTP_DRAFT_FILE}" ]; then
    DRAFT="$AEON_MTP_DRAFT_FILE"
    if [ ! -f "$MODELS_DIR/$DRAFT" ]; then
        RAW=$(cd "$MODELS_DIR" && find . -name '*assistant*4_*.gguf' ! -name '*.aeon.*' 2>/dev/null | head -1 | sed 's|^\./||')
        if [ -n "$RAW" ]; then
            echo "[adaptive-llamacpp] Normalizing MTP draft from $RAW ..."
            python3 "$SCRIPT_DIR/normalize_gemma4_assistant.py" "$MODELS_DIR/$RAW" "$MODELS_DIR/$DRAFT" \
                || { echo "[adaptive-llamacpp] WARN: draft normalize failed; MTP disabled"; DRAFT=""; }
        else
            echo "[adaptive-llamacpp] WARN: MTP draft $DRAFT missing and no raw assistant found; MTP disabled"
            DRAFT=""
        fi
    fi
fi

# Poll a container's /health. 0 healthy, 1 container died, 2 timeout.
wait_for_health() {
    local name=$1 port=$2 count=0
    while true; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then return 1; fi
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${port}/health 2>/dev/null || true)" = "200" ]; then return 0; fi
        sleep 5; count=$((count+1))
        [ $count -ge 180 ] && return 2
        [ $((count % 6)) -eq 0 ] && echo "[adaptive-llamacpp] ${name} still loading... ($((count*5))s)"
    done
}

# Launch one llama-server container. Args: name devices port ctx tensor_split ngl role
launch_node() {
    local name=$1 devices=$2 port=$3 ctx=$4 tensor_split=$5 ngl=$6 role=$7
    docker rm -f "$name" >/dev/null 2>&1 || true
    local args=(-m "/models/${TARGET}")
    if [ -n "$DRAFT" ]; then
        args+=(-md "/models/${DRAFT}" --spec-type draft-mtp --spec-draft-n-max "${AEON_MTP_NMAX:-6}" -ngld 99)
    fi
    args+=(-ngl "${NGL:-$ngl}" --flash-attn on -c "$ctx" --host 0.0.0.0 --port 8080 --threads "$PHYSICAL_CORES")
    [ -n "$KV_QUANT" ] && args+=(-ctk "$KV_QUANT" -ctv "$KV_QUANT")
    [ -n "$MMPROJ" ] && args+=(--mmproj "/models/${MMPROJ}")
    if [ -n "$CHAT_TEMPLATE" ]; then
        args+=(--chat-template-file "/models/${CHAT_TEMPLATE}")
        # --chat-template-file needs the jinja engine; the single role adds --jinja below.
        [ "$role" = "single" ] || args+=(--jinja)
    fi
    if [ "$role" = "single" ]; then
        # Split/offload: span GPUs, GPU0-weighted layer split, smaller ubatch for headroom.
        args+=(--split-mode layer --main-gpu 0 --tensor-split "${TENSOR_SPLIT:-$tensor_split}" \
               --batch-size "${BATCH:-2048}" --ubatch-size "${UBATCH:-512}" --jinja)
    else
        args+=(--batch-size "${BATCH:-4096}")
    fi
    echo "[adaptive-llamacpp] launch $name (GPU $devices, port $port, ctx $ctx${tensor_split:+, split $tensor_split})"
    docker run -d --label owner=aday --name "$name" --gpus "\"device=${devices}\"" -p "${port}:8080" \
        -v "${MODELS_DIR}:/models:ro" --shm-size=16g --ulimit memlock=-1 \
        "$IMAGE" "${args[@]}" > /tmp/aeon_${name}.cid 2>/tmp/aeon_${name}.err || {
            echo "[adaptive-llamacpp] docker run failed for $name:"; cat /tmp/aeon_${name}.err; return 1; }
}

# Launch with OOM-backoff: halve ctx (floor MIN_CTX) and retry if the node dies loading.
launch_with_backoff() {
    local name=$1 devices=$2 port=$3 ctx=$4 tensor_split=$5 ngl=$6 role=$7
    local try=0
    while :; do
        launch_node "$name" "$devices" "$port" "$ctx" "$tensor_split" "$ngl" "$role"
        wait_for_health "$name" "$port"; local rc=$?
        if [ $rc -eq 0 ]; then echo "[adaptive-llamacpp] $name READY (ctx $ctx)"; return 0; fi
        if [ $rc -eq 1 ] && [ $try -lt 2 ] && [ $((ctx/2)) -ge $MIN_CTX ]; then
            ctx=$((ctx/2)); try=$((try+1))
            echo "[adaptive-llamacpp] $name died (likely OOM); retrying at ctx $ctx"
            docker logs --tail 8 "$name" 2>&1 || true
            continue
        fi
        crash_log="/tmp/aeon_${name}.crash.log"
        echo "[adaptive-llamacpp] ERROR: $name failed to start (rc=$rc). Last 80 log lines (also saved to ${crash_log}):"
        # --tail must precede the container name (trailing flags are rejected); tee to a
        # host file so the reason survives the teardown that removes the container.
        docker logs --tail 80 "$name" 2>&1 | tee "$crash_log" || true
        return 1
    done
}

# --- Drive the plan ---
# Fields delimited by '|' (non-whitespace) so empty columns are NOT collapsed
# by `read` the way consecutive tabs/spaces would be.
NODE_LINES=$(python3 -c "
import os,json
p=json.loads(os.environ['AEON_DEPLOY_PLAN'])
for n in p['nodes']:
    print('|'.join(str(n.get(k,'')) for k in ('container','devices','port','ctx','tensor_split','ngl','role')))
")

NODE_URLS=()
echo "[adaptive-llamacpp] Cleaning up existing instances..."
while IFS='|' read -r name devices port ctx tensor_split ngl role; do
    [ -z "$name" ] && continue
    launch_with_backoff "$name" "$devices" "$port" "$ctx" "$tensor_split" "$ngl" "$role" || exit 1
    NODE_URLS+=("http://127.0.0.1:${port}")
done <<< "$NODE_LINES"

if [ "$TIER" = "dual" ]; then
    LB_NAME=$(pf "['container_name']")
    docker rm -f "$LB_NAME" >/dev/null 2>&1 || true
    JOINED=$(IFS=,; echo "${NODE_URLS[*]}")
    echo "[adaptive-llamacpp] Starting router $LB_NAME on :$LB_PORT -> $JOINED"
    docker run -d --label owner=aday --name "$LB_NAME" --network host \
        -e AEON_LB_NODES="$JOINED" -e AEON_LB_PORT="$LB_PORT" \
        -v "${SCRIPT_DIR}/adaptive_lb.py:/app/adaptive_lb.py" -w /app \
        python:3.11-slim sh -c "pip install -q fastapi uvicorn httpx && python adaptive_lb.py" > /dev/null
    # Wait for router
    for i in $(seq 1 30); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${LB_PORT}/health 2>/dev/null || true)" = "200" ] && break
        sleep 2
    done
fi

echo "[adaptive-llamacpp] ONLINE: ${AEON_SERVED_NAME} (tier $TIER) on http://localhost:${LB_PORT}"
