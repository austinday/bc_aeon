#!/usr/bin/env python3
"""
Quick diagnostic script to measure Qwen3.5 performance metrics.
Measures CPU RAM, GPU utilization, and tokens/second during inference.
"""

import subprocess
import time
import json
import requests
import psutil
import pynvml

def get_gpu_memory():
    """Get GPU memory usage for all GPUs."""
    try:
        pynvml.nvmlInit()
        result = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result.append({
                'gpu': i,
                'used_gb': info.used / (1024**3),
                'total_gb': info.total / (1024**3),
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            })
        pynvml.nvmlShutdown()
        return result
    except Exception as e:
        return [{'error': str(e)}]

def get_cpu_ram():
    """Get CPU RAM usage."""
    mem = psutil.virtual_memory()
    return {
        'used_gb': mem.used / (1024**3),
        'total_gb': mem.total / (1024**3),
        'percent': mem.percent
    }

def run_inference_benchmark(url, prompt_tokens=100, gen_tokens=50):
    """Run a quick inference benchmark and measure tokens/second."""
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "n_predict": gen_tokens,
                "temperature": 0.0,
                "stop": ["</s>"]
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            actual_tokens = len(data.get('content', '').split())
            tokens_per_sec = actual_tokens / elapsed if elapsed > 0 else 0
            return {
                'success': True,
                'elapsed_sec': elapsed,
                'tokens_generated': actual_tokens,
                'tokens_per_sec': tokens_per_sec,
                'prompt_tokens': prompt_tokens
            }
        else:
            return {'success': False, 'error': f'HTTP {response.status_code}', 'elapsed_sec': elapsed}
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Timeout', 'elapsed_sec': time.time() - start_time}
    except Exception as e:
        return {'success': False, 'error': str(e), 'elapsed_sec': time.time() - start_time}

def check_server_health(url):
    """Check if server is healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 70)
    print("QWEN3.5 PERFORMANCE DIAGNOSTIC")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {'name': 'Single GPU', 'url': 'http://localhost:8001', 'port': 8001},
        {'name': 'Dual GPU', 'url': 'http://localhost:8003', 'port': 8003}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} (port {config['port']}) ---")
        
        # Check if server is running
        if not check_server_health(config['url']):
            print(f"  Server not responding on port {config['port']}. Skipping.")
            results.append({
                'mode': config['name'],
                'status': 'server_not_running',
                'port': config['port']
            })
            continue
        
        print("  Server healthy. Collecting metrics...")
        
        # Get baseline GPU memory before inference
        gpu_before = get_gpu_memory()
        cpu_before = get_cpu_ram()
        
        # Run inference benchmark
        print("  Running inference benchmark (100 prompt tokens, 50 gen tokens)...")
        inference_result = run_inference_benchmark(config['url'])
        
        # Get GPU memory after inference
        gpu_after = get_gpu_memory()
        cpu_after = get_cpu_ram()
        
        result = {
            'mode': config['name'],
            'port': config['port'],
            'status': 'success' if inference_result.get('success') else 'failed',
            'inference': inference_result,
            'gpu_before': gpu_before,
            'gpu_after': gpu_after,
            'cpu_ram_before': cpu_before,
            'cpu_ram_after': cpu_after,
            'cpu_ram_delta_gb': (cpu_after['used_gb'] - cpu_before['used_gb'])
        }
        
        results.append(result)
        
        # Print summary
        print(f"  Inference: {inference_result.get('tokens_per_sec', 0):.2f} tokens/sec")
        print(f"  CPU RAM delta: {result['cpu_ram_delta_gb']:.2f} GB")
        for gpu in gpu_after:
            if 'gpu' in gpu:
                print(f"  GPU{gpu['gpu']}: {gpu['used_gb']:.1f}GB used, {gpu['utilization']}% util")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    single = next((r for r in results if r['mode'] == 'Single GPU'), None)
    dual = next((r for r in results if r['mode'] == 'Dual GPU'), None)
    
    if single and dual and single['status'] == 'success' and dual['status'] == 'success':
        single_tps = single['inference'].get('tokens_per_sec', 0)
        dual_tps = dual['inference'].get('tokens_per_sec', 0)
        
        if single_tps > 0:
            ratio = dual_tps / single_tps
            print(f"Single GPU: {single_tps:.2f} tokens/sec")
            print(f"Dual GPU:   {dual_tps:.2f} tokens/sec")
            print(f"Ratio (Dual/Single): {ratio:.2f}x")
            
            if ratio < 0.9:
                print("\n⚠️  WARNING: Dual GPU is SLOWER than single GPU!")
                print("   Possible causes:")
                print("   - CPU RAM thrashing (check CPU RAM delta)")
                print("   - PCIe bandwidth bottleneck")
                print("   - Improper layer splitting")
                print("   - Missing --mlock or --no-mmap flags")
            elif ratio > 1.1:
                print("\n✓ Dual GPU is performing as expected (faster than single)")
            else:
                print("\n⚠️  Dual GPU performance is similar to single (no significant gain)")
        
        # CPU RAM analysis
        print(f"\nCPU RAM Usage During Inference:")
        print(f"  Single GPU: {single['cpu_ram_delta_gb']:.2f} GB delta")
        print(f"  Dual GPU:   {dual['cpu_ram_delta_gb']:.2f} GB delta")
        
        if dual['cpu_ram_delta_gb'] > single['cpu_ram_delta_gb'] * 1.5:
            print("\n⚠️  WARNING: Dual GPU mode using significantly more CPU RAM!")
            print("   This indicates potential memory thrashing.")
    else:
        print("Could not compare - one or both servers not running")
        for r in results:
            print(f"  {r['mode']}: {r['status']}")
    
    # Save results to JSON
    with open('/home/aday/bc_aeon/aeon/scripts/diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: /home/aday/bc_aeon/aeon/scripts/diagnostic_results.json")

if __name__ == '__main__':
    main()