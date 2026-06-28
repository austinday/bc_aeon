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
    TOOL_DESC_BROWSER_SWITCH_TAB,
)

BROWSER_API_URL = "http://localhost:8030"

# Cap the structured element list so a huge page can't flood context. The agent
# can scroll to reveal more; function is primary but context still must survive.
MAX_ELEMENT_LINES = 160
MAX_ELEMENTS_CHARS = 14000


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
                line += f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m"
            print(line)
        print("\033[96m------------------------------\033[0m\n")
    except Exception as e:
        print(f"Failed to render image to terminal: {e}")


def _manage_browser_registry(action='register'):
    """Manage the current agent PID as an active user of the browser service."""
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

        cleaned_pids = []
        for p in active_pids:
            try:
                os.kill(p, 0)
                cleaned_pids.append(p)
            except OSError:
                pass

        if action == 'register':
            if pid not in cleaned_pids:
                cleaned_pids.append(pid)
        elif action == 'unregister':
            if pid in cleaned_pids:
                cleaned_pids.remove(pid)

        with open(registry_path, 'w') as f:
            json.dump(cleaned_pids, f)

        return len(cleaned_pids)


def ensure_browser_running():
    _manage_browser_registry('register')
    # Also register for the vision tool to keep it alive alongside the browser
    AnalyzeImageTool()._manage_registry('register')

    try:
        res = requests.get(f"{BROWSER_API_URL}/health", timeout=2)
        if res.status_code == 200:
            return True
    except Exception:
        pass

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_browser.sh"))
    subprocess.run(["bash", script_path], check=True)
    return True


def _session_id():
    return str(os.getpid())


def _format_elements(elements):
    """Render the stable, indexed element snapshot as readable text. This is the
    agent's primary, lossless view of what is on the page and actionable."""
    if not elements:
        return "(no interactive elements detected — the page may still be loading; try browser_read or scroll.)"

    on, off = [], []
    for e in elements:
        states = e.get("states") or []
        st = f" ({', '.join(states)})" if states else ""
        val = f" value='{e['value']}'" if e.get("value") else ""
        sc = f" {{scroll-group {e['scrollContainer']}}}" if e.get("scrollContainer") else ""
        fr = " «in iframe»" if e.get("inFrame") else ""
        name = e.get("name") or "(no text)"
        line = f"[{e['id']}] {e.get('role', '?')} \"{name}\"{val}{st}{sc}{fr}"
        (on if e.get("inViewport") else off).append(line)

    parts = []
    if on:
        parts.append("IN VIEW (visible now):\n" + "\n".join(on[:MAX_ELEMENT_LINES]))
    if off:
        shown = off[:max(0, MAX_ELEMENT_LINES - len(on))]
        if shown:
            parts.append("OFF-SCREEN (scroll to bring into view before interacting):\n" + "\n".join(shown))
    text = "\n\n".join(parts)
    if len(text) > MAX_ELEMENTS_CHARS:
        text = text[:MAX_ELEMENTS_CHARS] + "\n... [element list truncated — scroll or browser_read for a focused view] ..."
    total = len(elements)
    if total > len(on) + len(off[:max(0, MAX_ELEMENT_LINES - len(on))]):
        text += f"\n\n({total} interactive elements total; not all shown.)"
    return text


def _format_scroll(page_state):
    if not page_state:
        return ""
    try:
        y = page_state.get("scrollY", 0)
        sh = page_state.get("scrollHeight", 0)
        ch = page_state.get("clientHeight", 0)
        if sh <= ch:
            return "SCROLL: entire page fits in view (nothing more to scroll)."
        pct = int(100 * y / max(1, sh - ch))
        below = sh - (y + ch)
        more = []
        if y > 5:
            more.append("more above")
        if below > 5:
            more.append("more below")
        tail = f" ({', '.join(more)})" if more else ""
        return f"SCROLL: at ~{pct}% of page{tail}. Use scroll_down/scroll_up to reveal off-screen elements."
    except Exception:
        return ""


def process_browser_response(data, action_desc, session_id, tab_id, include_vision=True):
    if data.get("status") == "error":
        return f"Browser Error during {action_desc}: {data.get('msg')}"

    # Plain-text results (get_text / get_page_text) have no screenshot payload.
    if "clean_b64" not in data and "text" in data:
        return f"--- {action_desc} ---\nExtracted text:\n{data['text'][:8000]}"

    output_dir = os.path.expanduser(f"~/.aeon/temp/browser_output_{session_id}_{tab_id}")
    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "clean.jpg")
    overlay_path = os.path.join(output_dir, "overlay.jpg")

    clean_bytes = base64.b64decode(data["clean_b64"])
    with open(clean_path, "wb") as f:
        f.write(clean_bytes)
    with open(overlay_path, "wb") as f:
        f.write(base64.b64decode(data["overlay_b64"]))

    _print_image_to_terminal(clean_bytes, target_width=80)

    elements_str = _format_elements(data.get("elements", []))
    scroll_str = _format_scroll(data.get("page_state"))
    page_title = data.get("title", "Unknown")
    page_url = data.get("url", "Unknown")
    open_tabs = data.get("open_tabs", [])
    open_tabs_str = ", ".join(open_tabs) if open_tabs else tab_id

    # --- Vision channel (real Set-of-Mark overlay) ---
    # The full element text/roles/state are already in the structured list above,
    # so vision's job is only a FAST, concise visual check: layout + blockers +
    # what visibly changed. The short max_tokens keeps this off the critical path.
    if include_vision:
        vision_prompt = (
            "Numbered colored boxes mark interactive elements (each number is that element's id; "
            "the full element list is provided separately, so do NOT enumerate them). In 2-4 short "
            f"sentences: describe the visual layout, state what visibly changed after '{action_desc}', "
            "and explicitly flag any CAPTCHA / 'verify you are human' / cookie-or-consent wall / "
            "login prompt / loading spinner blocking the page. Be concise."
        )
        try:
            vision_analysis = AnalyzeImageTool().execute(
                image_path=overlay_path, prompt=vision_prompt, auto_cleanup=False,
                max_tokens=384, temperature=0.1,
            )
        except Exception as e:
            vision_analysis = f"Vision analysis unavailable: {e}"
    else:
        vision_analysis = "(vision skipped this step for speed; call browser_read with include_vision=true for a visual check.)"

    # Dialogs auto-handled and files downloaded since the last step are reported
    # here so the agent always knows they happened (and where downloads landed).
    events = data.get("events") or []
    events_str = ("\n=== EVENTS ===\n" + "\n".join(f"- {e}" for e in events) + "\n") if events else ""

    return (
        f"--- BROWSER: {action_desc} (tab '{tab_id}') ---\n"
        f"URL: {page_url}\n"
        f"Title: {page_title}\n"
        f"Open tabs: [{open_tabs_str}]\n"
        f"{scroll_str}\n"
        f"{events_str}\n"
        f"=== INTERACTIVE ELEMENTS (act on these by [id]) ===\n"
        f"{elements_str}\n\n"
        f"=== VISUAL ANALYSIS (Qwen-VL on the numbered screenshot) ===\n"
        f"{vision_analysis}\n\n"
        f"Screenshots: clean={clean_path} | numbered={overlay_path}"
    )


def _post(endpoint, payload, action_desc, tab_id, timeout=90, include_vision=True):
    try:
        ensure_browser_running()
        resp = requests.post(f"{BROWSER_API_URL}/{endpoint}", json=payload, timeout=timeout)
        if resp.status_code != 200:
            # The server returns a helpful 'detail' for 4xx (e.g. expected_text mismatch).
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return f"Browser action failed ({action_desc}): HTTP {resp.status_code}: {detail}"
        return process_browser_response(resp.json(), action_desc, payload.get("session_id"),
                                        tab_id, include_vision=include_vision)
    except Exception as e:
        return f"Error during {action_desc}: {type(e).__name__}: {e}"


class BrowserNavigateTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_navigate", description=TOOL_DESC_BROWSER_NAVIGATE)

    def execute(self, url: str, tab_id: str = "default", include_vision: bool = True, **kwargs) -> str:
        if not url:
            return "Error: 'url' is required."
        return _post("navigate", {"session_id": _session_id(), "tab_id": tab_id, "url": url},
                     f"Navigated to {url}", tab_id, include_vision=include_vision)


class BrowserInteractTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_interact", description=TOOL_DESC_BROWSER_INTERACT)

    def execute(self, action: str, element_id: int = None, text: str = None,
                expected_text: str = None, tab_id: str = "default", key: str = None,
                value: str = None, file_path: str = None, amount: int = None,
                direction: str = None, to_element_id: int = None,
                clear_first: bool = True, then_enter: bool = False,
                duration: int = 2000, include_vision: bool = True, **kwargs) -> str:
        if not action:
            return "Error: 'action' is required."
        # Friendly aliases so common phrasings just work.
        alias = {"scroll_down": ("scroll", "down"), "scroll_up": ("scroll", "up"),
                 "scroll_to_bottom": ("scroll", "bottom"), "scroll_to_top": ("scroll", "top"),
                 "enter": ("press_key", None)}
        if action in alias:
            mapped, d = alias[action]
            if mapped == "scroll" and direction is None:
                direction = d
            if action == "enter":
                key = key or "Enter"
            action = mapped

        payload = {
            "session_id": _session_id(), "tab_id": tab_id, "action": action,
            "element_id": element_id, "to_element_id": to_element_id, "text": text,
            "expected_text": expected_text, "key": key, "value": value,
            "file_path": file_path, "amount": amount, "direction": direction or "down",
            "clear_first": clear_first, "then_enter": then_enter, "duration": duration,
        }
        desc = f"{action}" + (f" on [{element_id}]" if element_id is not None else "")
        return _post("interact", payload, desc, tab_id, include_vision=include_vision)


class BrowserReadTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="browser_read",
            description=(
                "Re-observe the current page WITHOUT acting: returns a fresh indexed element list, "
                "scroll state, and a numbered screenshot. Use it after a page updates on its own, "
                "when element ids feel stale, or to re-orient before choosing the next action.\n"
                "Schema:\n  tab_id (str, optional, default='default'): the tab to read.\n"
                "  include_vision (bool, optional, default true): set false to skip the screenshot "
                "vision analysis for a faster read when the element list alone is enough.\n"
                "Example: {\"tool_name\": \"browser_read\", \"parameters\": {\"tab_id\": \"gmail\"}}"
            ),
        )

    def execute(self, tab_id: str = "default", include_vision: bool = True, **kwargs) -> str:
        return _post("observe", {"session_id": _session_id(), "tab_id": tab_id}, "Read page", tab_id,
                     include_vision=include_vision)


class BrowserCloseTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_close_tab", description=TOOL_DESC_BROWSER_CLOSE_TAB)

    def execute(self, tab_id: str) -> str:
        try:
            ensure_browser_running()
            resp = requests.post(f"{BROWSER_API_URL}/close_tab",
                                 json={"session_id": _session_id(), "tab_id": tab_id}, timeout=15)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            data = resp.json()
            remaining = data.get("remaining_tabs", 0)
            if remaining == 0:
                rem_browser = _manage_browser_registry('unregister')
                if rem_browser == 0:
                    subprocess.run(['docker', 'rm', '-f', 'aeon_browser'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rem_vision = AnalyzeImageTool()._manage_registry('unregister')
                if rem_vision == 0:
                    subprocess.run(['docker', 'rm', '-f', 'aeon_qwen36_vl'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Closed tab '{tab_id}'. No tabs left; released browser and vision resources."
            return f"Closed tab '{tab_id}'. {remaining} tab(s) still open."
        except Exception as e:
            return self.format_error_message(e, f"closing tab {tab_id}")


class BrowserSwitchTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_switch_tab", description=TOOL_DESC_BROWSER_SWITCH_TAB)

    def execute(self, tab_id: str = "default") -> str:
        return _post("switch_tab", {"session_id": _session_id(), "tab_id": tab_id},
                     f"Switched to tab '{tab_id}'", tab_id)
