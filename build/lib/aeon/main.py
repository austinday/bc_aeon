import os, argparse, json, time, sys, subprocess, requests, fcntl
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

LOCK_FILE_PATH = "/tmp/aeon_runtime.lock"

def is_container_running(name):
    try: return bool(subprocess.check_output(["docker", "ps", "-q", "-f", f"name={name}"], stderr=subprocess.DEVNULL, text=True).strip())
    except: return False

def wait_for_service(name, port):
    print(f"Waiting for {name} (Port {port})...", end='', flush=True)
    start = time.time()
    while time.time() - start < 60:
        try:
            if requests.get(f"http://localhost:{port}/api/tags", timeout=1).status_code == 200: 
                print(" OK.")
                return True
        except: pass
        time.sleep(1)
        print(".", end='', flush=True)
    print(" Timeout!")
    return False

def start_local_brain_services():
    if is_container_running("aeon_planner"):
        return
    print("\n[SYSTEM] Booting Local Brain...")
    script = Path(__file__).parent / "scripts" / "start_brain.sh"
    subprocess.run(["bash", str(script)], check=True)
    wait_for_service("Planner (DeepSeek)", 8000)
    wait_for_service("Executor (Qwen)", 8001)

def cleanup_transient_tools():
    print("[SYSTEM] Cleaning up transient tool containers...")
    try:
        subprocess.run("docker ps -a -q --filter 'name=aeon_research' --filter 'name=aeon_vision' | xargs -r docker rm -f", shell=True, stderr=subprocess.DEVNULL)
    except: pass

def unload_local_brain():
    print("[SYSTEM] Last agent exiting. Releasing Brain VRAM...")
    for port in [8000, 8001]:
        try: requests.post(f"http://localhost:{port}/api/generate", json={"model": "*", "keep_alive": 0}, timeout=2)
        except: pass

def get_ollama_models():
    try:
        resp = requests.get("http://localhost:8000/api/tags", timeout=1)
        if resp.status_code == 200:
            return sorted([m['name'] for m in resp.json().get('models', [])])
    except: pass
    return []

def select_model(models, label):
    print(f"\n[MENU] {label}")
    for i, m in enumerate(models):
        print(f" {i+1}. {m}")
    while True:
        try:
            choice = input(f"Select Model (1-{len(models)}): ")
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                return models[int(choice)-1]
        except (KeyboardInterrupt, EOFError): sys.exit(0)
        except: pass
        print("Invalid choice.")

class SessionManager:
    def __init__(self): self.lock = None
    def enter(self):
        self.lock = open(LOCK_FILE_PATH, 'w+')
        fcntl.flock(self.lock, fcntl.LOCK_SH)
    def exit(self):
        self.exit_brain()
    def exit_brain(self):
        cleanup_transient_tools()
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unload_local_brain()
        except BlockingIOError: pass
        finally: 
            try:
                fcntl.flock(self.lock, fcntl.LOCK_UN)
                self.lock.close()
            except: pass

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grok', action='store_true')
    parser.add_argument('--gemini', action='store_true')
    parser.add_argument('--gemini-flash', action='store_true')
    parser.add_argument('--local', action='store_true', help='Force local mode')
    parser.add_argument('--debug', action='store_true', help='Enable detailed LLM call logging to ~/')
    parser.add_argument('--strong', type=str, help='Model for Strong Node (Planner)')
    parser.add_argument('--weak', type=str, help='Model for Weak Node (Executor)')
    args = parser.parse_args()

    provider = "local"
    if args.grok: provider = "grok"
    elif args.gemini: provider = "gemini"
    elif args.gemini_flash: provider = "gemini-flash"

    local_strong = None
    local_weak = None

    if provider == "local":
        start_local_brain_services()
        if args.strong and args.weak:
            local_strong, local_weak = args.strong, args.weak
        else:
            models = get_ollama_models()
            if not models:
                 print("[WARN] No models found via API. Using defaults.")
                 models = ["deepseek-r1:70b", "qwen2.5:72b"]
            
            local_strong = args.strong if args.strong else select_model(models, "Select Strong Model (Planner)")
            local_weak = args.weak if args.weak else select_model(models, "Select Weak Model (Executor)")
            
        print(f"[CONFIG] Strong: {local_strong} | Weak: {local_weak}")

    session = SessionManager(); session.enter()
    try:
        llm_client = LLMClient(provider=provider, local_strong=local_strong, local_weak=local_weak)
        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        deps = {'llm_client': llm_client, 'worker': worker}
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)
        print(f"Aeon Ready (Mode: {provider.upper()}, Debug: {args.debug})")
        while True:
            try:
                obj = input("> ")
                if obj.strip(): 
                    if obj.strip() in ['exit', 'quit']: break
                    worker.run(obj)
            except (KeyboardInterrupt, EOFError): break
    finally: session.exit()

if __name__ == "__main__": cli()