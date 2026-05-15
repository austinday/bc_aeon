import asyncio
import httpx
import json
import time
import argparse
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class RequestResult:
    request_id: int
    ttft: float
    tps: float
    tokens: int
    duration: float
    success: bool

async def run_benchmark_request(client: httpx.AsyncClient, request_id: int, port: int, prompt: str, n_predict: int):
    url = f"http://localhost:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "temperature": 0.0
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    tokens_generated = 0
    
    try:
        async with client.stream("POST", url, json=payload, timeout=None) as response:
            if response.status_code != 200:
                return RequestResult(request_id, 0, 0, 0, 0, False)
            
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        
                        # Llama.cpp reports timings in the final chunk
                        if data.get('stop') and 'timings' in data:
                            # We use the server's reported tokens if available, otherwise our count
                            tokens_generated = data['timings'].get('predicted_n', tokens_generated)
                        
                        tokens_generated += 1
                    except json.JSONDecodeError:
                        pass
                        
        end_time = time.perf_counter()
        
        if first_token_time is None:
            return RequestResult(request_id, 0, 0, 0, 0, False)
            
        ttft = first_token_time - start_time
        gen_duration = end_time - first_token_time
        tps = tokens_generated / gen_duration if gen_duration > 0 else 0
        
        return RequestResult(request_id, ttft, tps, tokens_generated, end_time - start_time, True)
        
    except Exception as e:
        # print(f"Request {request_id} failed: {e}")
        return RequestResult(request_id, 0, 0, 0, 0, False)

async def main():
    parser = argparse.ArgumentParser(description="Enhanced MTP Benchmark with Concurrency")
    parser.add_argument("--port", type=int, default=8013, help="Port of the llama.cpp server")
    parser.add_argument("--prompt", type=str, default="Write a long and highly detailed essay about the history and future of artificial intelligence in space exploration:", help="Prompt")
    parser.add_argument("--n_predict", type=int, default=512, help="Tokens to predict")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--total_requests", type=int, default=1, help="Total requests to run")
    args = parser.parse_args()

    print(f"Starting Enhanced Benchmark...")
    print(f"Configuration: Port={args.port}, Concurrency={args.concurrency}, Total Requests={args.total_requests}, n_predict={args.n_predict}")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=None) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        
        async def sem_request(rid):
            async with semaphore:
                return await run_benchmark_request(client, rid, args.port, args.prompt, args.n_predict)

        start_wall_time = time.perf_counter()
        tasks = [sem_request(i) for i in range(args.total_requests)]
        results = await asyncio.gather(*tasks)
        end_wall_time = time.perf_counter()

    # Analysis
    successful_results = [r for r in results if r.success]
    num_success = len(successful_results)
    
    if num_success == 0:
        print("All requests failed. Check if the server is running.")
        return

    total_tokens = sum(r.tokens for r in successful_results)
    wall_duration = end_wall_time - start_wall_time
    aggregate_tps = total_tokens / wall_duration if wall_duration > 0 else 0
    
    avg_ttft = statistics.mean([r.ttft for r in successful_results])
    avg_tps = statistics.mean([r.tps for r in successful_results])
    
    print(f"\n{'='*20} RESULTS {'='*20}")
    print(f"Total Requests:    {args.total_requests}")
    print(f"Successful:        {num_success}")
    print(f"Total Tokens:       {total_tokens}")
    print(f"Wall Clock Time:    {wall_duration:.3f}s")
    print(f"Aggregate Throughput: {aggregate_tps:.2f} tokens/s")
    print(f"Average TTFT:      {avg_ttft:.3f}s")
    print(f"Average Request TPS: {avg_tps:.2f} tokens/s")
    print(f"{'='*51}\n")

    if args.total_requests > 1:
        print("Per-request breakdown:")
        for r in successful_results:
            print(f"Req {r.request_id}: TTFT={r.ttft:.3f}s, TPS={r.tps:.2f}, Tokens={r.tokens}")

if __name__ == "__main__":
    asyncio.run(main())