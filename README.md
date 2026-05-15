# MTP Benchmark Results

The project includes an enhanced benchmark script to measure the performance of the llama.cpp server with Multi-Token Prediction (MTP).

## Benchmark Script
`scripts/benchmark_mtp_enhanced.py`

This script supports:
- **Single Agent Generation**: Measure TTFT and TPS for a single request.
- **Batch/Parallel Generation**: Measure aggregate throughput (total tokens/s) across multiple concurrent requests.

## Performance Results
Running the benchmark on the current setup:
- **Configuration**: Port 8013, Concurrency 4, Total Requests 8, `n_predict` 512.
- **Aggregate Throughput**: 116.69 tokens/s
- **Average TTFT**: 0.741s
- **Average Request TPS**: 38.52 tokens/s

## Usage
To run the benchmark:
```bash
python3 scripts/benchmark_mtp_enhanced.py --port 8013 --concurrency 4 --total_requests 8
```