import time
import requests
import json
import argparse

def benchmark(port, prompt, n_predict):
    url = f"http://localhost:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "temperature": 0.0
    }
    
    print(f"Benchmarking server on port {port}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print(f"Make sure the llama.cpp server is running on port {port}.")
        return

    first_token_time = None
    tokens_generated = 0
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                try:
                    data = json.loads(data_str)
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    # Llama.cpp usually reports exact timings in the final chunk
                    if data.get('stop') and 'timings' in data:
                        timings = data['timings']
                        tps = timings.get('predicted_per_second', 0)
                        predicted = timings.get('predicted_n', tokens_generated)
                        ttft_val = first_token_time - start_time
                        print(f"TTFT:{ttft_val:.3f}s Gen:N/A Tool:N/A Load:N/A TPS:{tps:.2f} Tokens:{predicted}")
                        return
                    
                    tokens_generated += 1
                except:
                    pass

    end_time = time.time()
    
    if first_token_time is None:
        print("No tokens received.")
        return
        
    ttft = first_token_time - start_time
    gen_time = end_time - first_token_time
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"TTFT:{ttft:.3f}s Gen:{gen_time:.3f}s Tool:N/A Load:N/A TPS:{tps:.2f} Tokens:{tokens_generated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MTP/Speculative Decoding generation speed")
    parser.add_argument("--port", type=int, default=8013, help="Port of the llama.cpp server (e.g. 8013 for MTP, 8006 for baseline)")
    parser.add_argument("--prompt", type=str, default="Write a long and highly detailed essay about the history and future of artificial intelligence in space exploration:", help="Prompt to trigger sustained generation")
    parser.add_argument("--n_predict", type=int, default=512, help="Number of tokens to predict")
    args = parser.parse_args()
    
    benchmark(args.port, args.prompt, args.n_predict)
