#!/bin/bash
# =============================================================================
# Qwen3.5-397B Performance Benchmark Script
# =============================================================================
# Measures CPU RAM, GPU utilization, and inference speed for single vs dual GPU
# Usage: bash benchmark_qwen.sh [single|dual] [num_requests]
# =============================================================================
set -e

MODE=${1:-single}
NUM_REQUESTS=${2:-5}

if [ "$MODE" = "single" ]; then
    PORT=8001
    CONTAINER_NAME='aeon_qwen397b'
    echo "=========================================="
    echo "BENCHMARK: Single GPU Mode"
    echo "=========================================="
elif [ "$MODE" = "dual" ]; then
    PORT=8003
    CONTAINER_NAME='aeon_qwen397b_dual'
    echo "=========================================="
    echo "BENCHMARK: Dual GPU Mode"
    echo "=========================================="
else
    echo "Usage: bash benchmark_qwen.sh [single|dual] [num_requests]"
    exit 1
fi

# Function to get GPU memory usage
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}'
}

# Function to get CPU RAM usage
get_cpu_ram() {
    free -g | awk '/^Mem:/ {print $3}'
}

# Function to run inference benchmark
run_benchmark() {
    local start_time=$(date +%s.%N)
    local total_tokens=0
    
    echo "Running $NUM_REQUESTS inference requests..."
    
    for i in $(seq 1 $NUM_REQUESTS); do
        local req_start=$(date +%s.%N)
        
        # Simple completion request
        response=$(curl -s -X POST "http://localhost:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "The quick brown fox",
                "n_predict": 128,
                "temperature": 0.7
            }' 2>/dev/null || echo '{"error": "failed"}')
        
        local req_end=$(date +%s.%N)
        local req_duration=$(echo "$req_end - $req_start" | bc)
        
        # Extract token count from response if available
        tokens=$(echo "$response" | grep -o '"tokens": [0-9]*' | grep -o '[0-9]*' || echo "0")
        total_tokens=$((total_tokens + tokens))
        
        echo "  Request $i: ${req_duration}s (tokens: $tokens)"
    done
    
    local end_time=$(date +%s.%N)
    local total_duration=$(echo "$end_time - $start_time" | bc)
    local avg_duration=$(echo "scale=3; $total_duration / $NUM_REQUESTS" | bc)
    
    echo ""
    echo "=========================================="
    echo "BENCHMARK RESULTS"
    echo "=========================================="
    echo "Total requests: $NUM_REQUESTS"
    echo "Total duration: ${total_duration}s"
    echo "Average per request: ${avg_duration}s"
    echo "Total tokens generated: $total_tokens"
    if [ "$total_tokens" -gt 0 ]; then
        local tokens_per_sec=$(echo "scale=2; $total_tokens / $total_duration" | bc)
        echo "Tokens per second: $tokens_per_sec"
    fi
}

# Check if container is running
echo "Checking container status..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container $CONTAINER_NAME is not running."
    echo "Please start the server first with:"
    if [ "$MODE" = "single" ]; then
        echo "  bash aeon/scripts/start_qwen397b.sh"
    else
        echo "  bash aeon/scripts/start_qwen397b_dual.sh"
    fi
    exit 1
fi

# Wait for health check
echo "Waiting for server health check..."
count=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server is healthy (HTTP $HTTP_CODE)"
        break
    fi
    sleep 2
    count=$((count+1))
    if [ $count -ge 30 ]; then
        echo "ERROR: Server did not become healthy within 60 seconds."
        exit 1
    fi
done

# Collect baseline metrics
echo ""
echo "Collecting baseline system metrics..."
baseline_gpu_mem=$(get_gpu_memory)
baseline_cpu_ram=$(get_cpu_ram)
echo "Baseline GPU memory (all GPUs): ${baseline_gpu_mem}MB"
echo "Baseline CPU RAM: ${baseline_cpu_ram}GB"

# Run benchmark
echo ""
run_benchmark

# Collect post-benchmark metrics
echo ""
echo "Collecting post-benchmark system metrics..."
post_gpu_mem=$(get_gpu_memory)
post_cpu_ram=$(get_cpu_ram)
echo "Post-benchmark GPU memory (all GPUs): ${post_gpu_mem}MB"
echo "Post-benchmark CPU RAM: ${post_cpu_ram}GB"

# Calculate deltas
gpu_delta=$((post_gpu_mem - baseline_gpu_mem))
ram_delta=$((post_cpu_ram - baseline_cpu_ram))
echo ""
echo "=========================================="
echo "RESOURCE DELTA"
echo "=========================================="
echo "GPU memory change: ${gpu_delta}MB"
echo "CPU RAM change: ${ram_delta}GB"

echo ""
echo "Benchmark complete for $MODE mode."