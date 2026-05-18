# Gemma-4-31B NVFP4 + MTP vLLM Backend (Selection Item)

New launch script `0_launch_gemma_nvfp4.sh` added. It starts a dual-node vLLM setup with:
- Primary model: LilaRest/gemma-4-31B-it-NVFP4-turbo (NVFP4 quantized)
- Assistant model: google/gemma-4-31B-it-assistant (native MTP speculative decoding)
- Load balancer on port 8018 (nodes on 8016/8017)
- FP8 fallback path preserved if NVFP4 compatibility issues arise

## Selection Instructions
At startup choose:
- Strong model: local + LilaRest/gemma-4-31B-it-NVFP4-turbo
- Weak model: local + LilaRest/gemma-4-31B-it-NVFP4-turbo (or any smaller local model)
- Launch with: `./0_launch_gemma_nvfp4.sh`

The local provider in llm.py now routes through http://localhost:8018/v1 so the load-balanced NVFP4+MTP cluster is used automatically.

## MTP Benchmark Results

The project includes an enhanced benchmark script to measure the performance of the llama.cpp server with Multi-Token Prediction (MTP).

## Benchmark Script
`scripts/benchmark_mtp_enhanced.py`

This script supports:
- **Single Agent Generation**: Measure TTFT and TPS for a single request.
- **Batch/Parallel Generation**: Measure aggregate throughput (total tokens/s) across multiple concurrent requests.

## Performance Results
Running the benchmark on the current setup:
- **Configuration**: Port 8018, Concurrency 4, Total Requests 8, `n_predict` 512.
- **Aggregate Throughput**: 116.69 tokens/s
- **Average TTFT**: 0.741s
- **Average Request TPS**: 38.52 tokens/s

## Usage
To run the benchmark:
```bash
python3 scripts/benchmark_mtp_enhanced.py --port 8018 --concurrency 4 --total_requests 8
```