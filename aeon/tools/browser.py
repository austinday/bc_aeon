import os
import base64
import requests
import subprocess
import json
import fcntl
import io
from PIL import Image
from .base import BaseTool
from .vision import AnalyzeImageTool
from ..core.prompts import (
    TOOL_DESC_BROWSER_NAVIGATE,
    TOOL_DESC_BROWSER_INTERACT,
    TOOL_DESC_BROWSER_CLOSE_TAB,
    TOOL_DESC_BROWSER_SWITCH_TAB
)

BROWSER_API_URL = "http://localhost:8030"

def _print_image_to_terminal(image_bytes, target_width=80):
    """Renders an image directly in the terminal using ANSI truecolor and half-block characters."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio / 2)
        
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        
        print("\n\033[96m--- Browser Vision Preview ---\033[0m")
        for y in range(0, target_height, 2):
            line = ""
            for x in range(target_width):
                r1, g1, b1 = img.getpixel((x, y))
                if y + 1 < target_height:
                    r2, g2, b2 = img.getpixel((x, y + 1))
                else:
                    r2, g2, b2 = (0, 0, 0)
                line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m\u2580\033[0m"
            print(line)
        print("\033[96m------------------------------\033[0m\n")
    except Exception as e:
        print(f"Failed to render image to terminal: {e}")

def _manage_browser_registry():
    """Register the current agent PID as an active user of the browser service."""
    registry_path = "/tmp/aeon_browser_registry.json"
    lock_path = "/tmp/aeon_browser_registry.lock"
    pid = os.getpid()
    
    with open(lock_path, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            active_pids = []
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    active_pids = json.load(f)
        except (json.JSONDecodeError, EOFError):
            active_pids = []
            
        # Clean up dead PIDs to keep registry tidy
        cleaned_pids = []
        for p in active_pids:
            try:
                os.kill(p, 0)
                cleaned_pids.append(p)
            except OSError:
                pass
        
        if pid not in cleaned_pids:
            cleaned_pids.append(pid)
            
        with open(registry_path, 'w') as f:
            json.dump(cleaned_pids, f)

def ensure_browser_running():
    _manage_browser_registry()
    try:
        res = requests.get(f"{BROWSER_API_URL}/health", timeout=2)
        if res.status_code == 200:
            return True
    except:
        pass
        
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_browser.sh"))
    subprocess.run(["bash", script_path], check=True)
    return True

def process_browser_response(data, action_desc, session_id, tab_id):
    if data.get("status") == "error":
        return f"Browser Error during {action_desc}: {data.get('msg')}"
        
    # Save screenshots in isolated folders per session and tab to avoid overwrites
    output_dir = os.path.expanduser(f"~/.aeon/temp/browser_output_{session_id}_{tab_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    clean_path = os.path.join(output_dir, "clean.jpg")
    overlay_path = os.path.join(output_dir, "overlay.jpg")
    
    clean_bytes = base64.b64decode(data["clean_b64"])
    with open(clean_path, "wb") as f:
        f.write(clean_bytes)
        
    with open(overlay_path, "wb") as f:
        f.write(base64.b64decode(data["overlay_b64"]))
        
    # Inject terminal render right as the tool receives the payload
    _print_image_to_terminal(clean_bytes, target_width=80)
        
    elements = data.get("elements", [])
    element_str = "\n".join([f"[{el['id']}] <{el['tag']}>: {el['text']}" for el in elements])
    
    markdown = data.get("markdown", "")
    
    # --- AUTO-VISION INJECTION ---
    vision_tool = AnalyzeImageTool()
    # ROBUST VISION PROMPT: Forces Qwen-VL to map products to specific IDs
    vision_prompt = (
        "This is a webpage with numbered red bounding boxes over interactive elements. "
        "Perform a detailed visual mapping:\n"
        "1. Describe the overall layout.\n"
        "2. Explicitly list all visible products, distinct images, or main content blocks and state their EXACT corresponding red box numbers.\n"
        "3. Identify key navigation links and search bars with their numbers.\n"
        "Be highly specific. If there are multiple 'View Products' buttons, clarify which number belongs to which product."
    )
    try:
        vision_analysis = vision_tool.execute(image_path=overlay_path, prompt=vision_prompt)
    except Exception as e:
        vision_analysis = f"Vision analysis failed: {e}"
    
    result = (
        f"--- BROWSER ACTION SUCCESS: {action_desc} (Tab: '{tab_id}') ---\n\n"
        f"--- VISUAL LAYOUT ANALYSIS (from Qwen-VL) ---\n"
        f"{vision_analysis}\n\n"
        f"--- VISIBLE TEXT (MARKDOWN) ---\n"
        f"{markdown}\n\n"
        f"--- INTERACTIVE ELEMENTS IN VIEWPORT ---\n"
        f"{element_str if element_str else 'No interactive elements visible.'}\n\n"
        f"Screenshots saved locally to:\n"
        f"  Clean:   {clean_path}\n"
        f"  Overlay: {overlay_path}"
    )
    return result

class BrowserNavigateTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_navigate", description=TOOL_DESC_BROWSER_NAVIGATE)
        
    def execute(self, url: str, tab_id: str = "default") -> str:
        if not url.startswith("http"):
            url = "https://" + url
        try:
            ensure_browser_running()
            session_id = str(os.getpid())
            resp = requests.post(f"{BROWSER_API_URL}/navigate", json={"url": url, "session_id": session_id, "tab_id": tab_id}, timeout=60)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            return process_browser_response(resp.json(), f"Navigated to {url}", session_id, tab_id)
        except Exception as e:
            return self.format_error_message(e, f"navigating to {url} in tab '{tab_id}'")

class BrowserInteractTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_interact", description=TOOL_DESC_BROWSER_INTERACT)
        
    def execute(self, action: str, element_id: int = None, text: str = None, expected_text: str = None, tab_id: str = "default") -> str:
        try:
            ensure_browser_running()
            session_id = str(os.getpid())
            payload = {"action": action, "session_id": session_id, "tab_id": tab_id}
            if element_id is not None:
                payload["element_id"] = element_id
            if text is not None:
                payload["text"] = text
            if expected_text is not None:
                payload["expected_text"] = expected_text
                
            resp = requests.post(f"{BROWSER_API_URL}/interact", json=payload, timeout=60)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            return process_browser_response(resp.json(), f"Action '{action}' on ID {element_id}", session_id, tab_id)
        except Exception as e:
            return self.format_error_message(e, f"performing {action} on element {element_id} in tab '{tab_id}'")

class BrowserCloseTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_close_tab", description=TOOL_DESC_BROWSER_CLOSE_TAB)
        
    def execute(self, tab_id: str) -> str:
        try:
            ensure_browser_running()
            session_id = str(os.getpid())
            resp = requests.post(f"{BROWSER_API_URL}/close_tab", json={"session_id": session_id, "tab_id": tab_id}, timeout=10)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            return f"Successfully closed tab: {tab_id}"
        except Exception as e:
            return self.format_error_message(e, f"closing tab {tab_id}")

class BrowserSwitchTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_switch_tab", description=TOOL_DESC_BROWSER_SWITCH_TAB)
        
    def execute(self, tab_id: str = "default") -> str:
        try:
            ensure_browser_running()
            session_id = str(os.getpid())
            resp = requests.post(f"{BROWSER_API_URL}/switch_tab", json={"session_id": session_id, "tab_id": tab_id}, timeout=60)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            return process_browser_response(resp.json(), f"Switched to tab '{tab_id}'", session_id, tab_id)
        except Exception as e:
            return self.format_error_message(e, f"switching to tab {tab_id}")
