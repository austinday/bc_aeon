import os
import shutil
import time
import json
import subprocess
import uuid
import sys
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from .base import BaseTool
from ..core.worker import Worker
from ..core.llm import LLMClient

class DockerExecTool(BaseTool):
    """A tool that replaces RunCommandTool for Sub-Agents, executing inside a Docker container."""
    def __init__(self, container_name: str):
        super().__init__(
            name="run_command",
            description='Executes shell commands INSIDE the isolated research container. Params: `command` (str).'
        )
        self.container_name = container_name

    def execute(self, command: str, timeout: int = 300):
        docker_cmd = f"docker exec -w /workspace {self.container_name} bash -c {json.dumps(command)}"
        try:
            process = subprocess.run(
                docker_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = process.stdout + process.stderr
            if process.returncode != 0:
                return f"COMMAND FAILED (Exit Code {process.returncode})\n\nOUTPUT:\n{output}"
            return f"COMMAND SUCCESS\n\nOUTPUT:\n{output}"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error running docker exec: {e}"

class SubmitFindingsTool(BaseTool):
    """Tool for research agents to return data and exit."""
    def __init__(self):
        super().__init__(
            name="submit_findings",
            description='Submit your research results and EXIT. Params: `findings` (high-level human-readable interpretation/results, NO CODE DUMP), `summary` (concise 1-sentence conclusion for the user).'
        )
    def execute(self, findings: str, summary: str):
        return f"Findings Submitted.\nSUMMARY: {summary}\nDETAILS:\n{findings}"

class ConductResearchTool(BaseTool):
    """A tool to perform parallel experiments in isolated Docker containers."""
    
    def __init__(self, llm_client: LLMClient, worker: Worker):
        super().__init__(
            name="conduct_research",
            description='Spawns parallel "Research Assistant" agents in ISOLATED DOCKER CONTAINERS. Use this to test conflicting libraries, destructive changes, or distinct hypotheses safely. Returns a report. Params: `topic` (str), `hypotheses` (List[{"name": str, "goal": "Detailed instruction..."}]), `image_tag` (optional str), `iterations` (optional int, default 30).'
        )
        self.llm_client = llm_client
        self.main_worker = worker
        self.host_labs_root = ".aeon_labs"

    def _setup_docker_lab(self, lab_id: str, image_tag: str, mem_limit_gb: float) -> Dict[str, str]:
        host_dir = os.path.abspath(os.path.join(self.host_labs_root, lab_id))
        if os.path.exists(host_dir):
            shutil.rmtree(host_dir)
        os.makedirs(host_dir)
        
        source_root = os.path.abspath(".")
        heavy_dirs = {'node_modules', 'venv', '.git', '__pycache__', 'data', 'target', 'build', 'dist', 'site-packages', '.aeon_labs'}
        
        for item in os.listdir(source_root):
            if item == ".aeon_labs": continue
            s = os.path.join(source_root, item)
            d = os.path.join(host_dir, item)
            if os.path.isdir(s):
                if item not in heavy_dirs and not item.startswith('.'):
                    shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
                
        container_name = f"aeon_lab_{lab_id}"
        mounts = f"-v {host_dir}:/workspace"
        data_path = os.path.join(source_root, 'data')
        if os.path.exists(data_path):
             mounts += f" -v {data_path}:/workspace/data:ro"
        
        # Enforce memory limit
        mem_arg = f"--memory={int(mem_limit_gb * 1024)}m"
        
        run_cmd = (f"docker run -d --rm --name {container_name} {mounts} {mem_arg} --cpus=1.0 -w /workspace {image_tag} tail -f /dev/null") 
        subprocess.run(run_cmd, shell=True, check=True)
        time.sleep(2)
        return {"container_name": container_name, "host_path": host_dir}

    def _run_researcher(self, hypothesis: Dict[str, Any], image_tag: str, iterations: int, mem_limit: float, progress_dict: Dict, printer_log: List) -> Dict[str, Any]:
        name = hypothesis.get("name", "unknown")
        goal = hypothesis.get("goal", "")
        lab_id = f"{str(uuid.uuid4())[:8]}_{''.join(x for x in name if x.isalnum())}"
        
        def silent_print(msg):
            printer_log.append(f"[{name}] {msg}")

        def update_progress(iteration, max_iter, status):
            progress_dict[name] = {"iter": iteration, "max": max_iter, "status": status}

        container_info = {}
        try:
            update_progress(0, iterations, "Booting")
            container_info = self._setup_docker_lab(lab_id, image_tag, mem_limit)
            container_name = container_info['container_name']
            host_path = container_info['host_path']

            sub_worker = Worker(llm_client=self.llm_client, print_func=silent_print)
            
            from .loader import load_tools_from_directory
            deps = {'llm_client': self.llm_client, 'worker': sub_worker, 'max_tokens': 20000}
            all_tools = load_tools_from_directory("aeon.tools", deps, verbose=False)
            
            research_tools = []
            for t in all_tools:
                if t.name in ["conduct_research", "dispatch_subtasks", "run_command", "say_to_user", "get_user_input", "task_complete"]:
                    continue
                if t.name == "write_file": continue 
                research_tools.append(t)
            
            research_tools.append(DockerExecTool(container_name))
            research_tools.append(SubmitFindingsTool()) 
            
            class HostPathWriteTool(BaseTool):
                def __init__(self, target_dir):
                    super().__init__("write_file", "Writes file to the research lab.")
                    self.target_dir = target_dir
                def execute(self, file_path, content):
                    if file_path.startswith("/"): file_path = file_path.lstrip("/")
                    full_path = os.path.join(self.target_dir, file_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f: f.write(content)
                    return f"Written to {file_path} (inside container)"
            
            research_tools.append(HostPathWriteTool(host_path))
            sub_worker.register_tools(research_tools)

            # --- SUB-AGENT DIRECTIVES ---
            sub_worker.base_directives = (
                f"ROLE: Researcher in DOCKER container {container_name}.\n"
                f"HYPOTHESIS: {name}\nGOAL: {goal}\n"
                f"RESOURCES: You have approx {mem_limit:.1f}GB RAM and 1 CPU core. You share 1 GPU with others.\n"
                f"1. RESOURCE AWARENESS: Use small batch sizes. Do NOT hog the GPU. Prefer CPU if possible.\n"
                f"2. INSTALLATION: `pip install` works. Do NOT use Docker (you are in it).\n"
                f"3. REPORTING: Use `submit_findings`. Do NOT include code. Provide a high-level, human-readable interpretation of the results and what was learned.\n"
                f"4. CLEANUP: Your lab will be deleted.\n"
            )

            update_progress(0, iterations, "Starting")
            
            sub_worker.run(
                f"Test Hypothesis '{name}': {goal}", 
                max_iterations=iterations, 
                step_callback=update_progress,
                terminal_tools=["submit_findings"]
            )
            
            final_report = "No report."
            status = "Incomplete"
            if sub_worker.recent_history:
                last = sub_worker.recent_history[-1]
                if last['action'] == "submit_findings":
                    final_report = last['summary']
                    status = "Complete"
                else:
                    final_report = f"Timed out. Last action: {last['action']}"
            
            update_progress(iterations, iterations, "Finished")
            return {"name": name, "status": status, "report": final_report, "lab_id": lab_id, "host_path": host_path}

        except Exception as e:
            update_progress(0, 0, "Error")
            return {"name": name, "status": "Error", "report": str(e)}
        finally:
            if container_info.get('container_name'):
                subprocess.run(f"docker kill {container_info['container_name']}", shell=True, stderr=subprocess.DEVNULL)
            if container_info.get('host_path') and os.path.exists(container_info['host_path']):
                shutil.rmtree(container_info['host_path'])

    def execute(self, topic: str, hypotheses: List[Dict[str, str]], image_tag: str = "aeon_base:py3.10-cuda12.1", iterations: int = 30) -> str:
        results = []
        futures = {}
        printer_logs = {} 
        
        # --- RESOURCE CALCULATION ---
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        available_mem = total_mem_gb * 0.8
        num_threads = min(len(hypotheses), 10)
        mem_per_agent = available_mem / num_threads

        progress_data = {h['name']: {"iter": 0, "max": iterations, "status": "Queued"} for h in hypotheses}
        
        print(f"\n{BaseTool.C_CYAN}=== RESEARCH DASHBOARD: {topic} ==={BaseTool.C_RESET}")
        print(f"Image: {image_tag} | Threads: {num_threads} | Limit: {mem_per_agent:.1f}GB/Agent\n")
        
        for _ in hypotheses:
            print("") 

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for hypo in hypotheses:
                name = hypo['name']
                printer_logs[name] = []
                futures[executor.submit(self._run_researcher, hypo, image_tag, iterations, mem_per_agent, progress_data, printer_logs[name])] = name
            
            all_done = False
            first_run = True
            
            while not all_done:
                if all(f.done() for f in futures):
                    all_done = True
                
                if not first_run:
                    sys.stdout.write(f"\033[{len(hypotheses)}A")
                    sys.stdout.flush()
                
                for hypo in hypotheses:
                    name = hypo['name']
                    data = progress_data.get(name, {})
                    cur = data.get('iter', 0)
                    mx = data.get('max', iterations)
                    stat = data.get('status', 'Init')
                    
                    bar_len = 20
                    filled = int((cur / mx) * bar_len) if mx > 0 else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    
                    color = BaseTool.C_YELLOW
                    if stat == "Complete": color = BaseTool.C_GREEN
                    elif stat == "Error": color = BaseTool.C_RED
                    
                    print(f"\033[K{color}[{stat:^10}] {name[:25]:<25} {bar} {cur}/{mx}{BaseTool.C_RESET}")
                
                first_run = False
                if not all_done:
                    time.sleep(0.5)
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"name": futures[future], "status": "Crash", "report": str(e)})
        
        print(f"\n{BaseTool.C_GREEN}=== Research Completed ==={BaseTool.C_RESET}")
        report = [f"# Research Summary: {topic}"]
        
        for res in results:
            summary_text = "(No summary provided)"
            full_report = res.get('report', '')
            if "SUMMARY:" in full_report:
                try:
                    parts = full_report.split("SUMMARY:", 1)
                    if len(parts) > 1:
                        summary_text = parts[1].split("DETAILS:", 1)[0].strip()
                except Exception:
                    pass
            
            print(f"- {BaseTool.C_CYAN}{res['name']}:{BaseTool.C_RESET} {summary_text}")
            report.append(f"## Hypothesis: {res['name']} ({res['status']})")
            report.append(f"{full_report}\n")
            
        return "\n".join(report)
