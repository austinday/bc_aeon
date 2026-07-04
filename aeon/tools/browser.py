import os
import re
import sys
import base64
import requests
import subprocess
import json
import fcntl
import io
import shutil
from PIL import Image
from .base import BaseTool
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
    # Only worth doing when a human is watching an interactive terminal. In an
    # autonomous/sub-agent run (piped stdout) it just burns CPU on a PIL resize +
    # per-pixel loop and floods the logs with escape codes, so skip it entirely.
    if not sys.stdout.isatty():
        return
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


def _browser_healthy():
    try:
        return requests.get(f"{BROWSER_API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


_pruned_stale_output = False


def _prune_stale_output_dirs():
    """Remove ~/.aeon/temp/browser_output_<pid>_<tab> dirs left by agent processes
    that are no longer alive, so screenshots don't accumulate on disk across runs.
    Runs once per process and only ever deletes dirs whose owning PID is dead."""
    global _pruned_stale_output
    if _pruned_stale_output:
        return
    _pruned_stale_output = True
    base = os.path.expanduser("~/.aeon/temp")
    try:
        if not os.path.isdir(base):
            return
        my_pid = os.getpid()
        for name in os.listdir(base):
            if not name.startswith("browser_output_"):
                continue
            parts = name.split("_")  # browser, output, <pid>, <tab...>
            if len(parts) < 4 or not parts[2].isdigit():
                continue
            pid = int(parts[2])
            if pid == my_pid:
                continue
            try:
                os.kill(pid, 0)          # exists (alive, or ours) -> keep
            except ProcessLookupError:
                shutil.rmtree(os.path.join(base, name), ignore_errors=True)
            except OSError:
                pass                     # exists but not signalable -> keep
    except Exception:
        pass


def ensure_browser_running():
    _manage_browser_registry('register')
    _prune_stale_output_dirs()
    if _browser_healthy():
        return True

    # Serialize startup ACROSS agent processes: without this, the principal and
    # its sub-agents can each fire `docker run` at once and collide on the
    # container name. Hold an exclusive lock, re-check health inside it (another
    # process may have just started the service), then start exactly once.
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_browser.sh"))
    with open("/tmp/aeon_browser_start.lock", "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if _browser_healthy():
            return True
        subprocess.run(["bash", script_path], check=True)
    return True


def _session_id():
    return str(os.getpid())


def _profile_for(worker):
    """The browser profile (isolation unit) for this agent. The principal uses
    'default' (shared, persistent — logins survive); each sub-agent sets its own
    on the worker so it browses as an independent identity."""
    return getattr(worker, "browser_profile", "default") or "default"


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
        x = page_state.get("scrollX", 0)
        sw = page_state.get("scrollWidth", 0)
        cw = page_state.get("clientWidth", 0)
        vscroll = sh > ch
        hscroll = sw > cw
        if not vscroll and not hscroll:
            return "SCROLL: entire page fits in view (nothing more to scroll)."
        parts = []
        if vscroll:
            pct = int(100 * y / max(1, sh - ch))
            more = []
            if y > 5:
                more.append("more above")
            if sh - (y + ch) > 5:
                more.append("more below")
            tail = f" ({', '.join(more)})" if more else ""
            parts.append(f"vertical ~{pct}%{tail}")
        if hscroll:
            hpct = int(100 * x / max(1, sw - cw))
            hmore = []
            if x > 5:
                hmore.append("more left")
            if sw - (x + cw) > 5:
                hmore.append("more right")
            htail = f" ({', '.join(hmore)})" if hmore else ""
            parts.append(f"horizontal ~{hpct}%{htail}")
        hint = ("Use scroll direction down/up" + ("/left/right" if hscroll else "")
                + " (optionally with element_id to scroll a specific pane).")
        return f"SCROLL: {'; '.join(parts)}. {hint}"
    except Exception:
        return ""


def _relocate_downloads(events):
    """Copy any files the browser just downloaded (reported in `events`, saved
    inside the browser profile) into the agent's WORKSPACE ./downloads so the
    agent can actually use them. Returns extra event lines noting the new paths."""
    extra = []
    for e in events or []:
        if not e.startswith("[download] saved"):
            continue
        m = re.search(r"-> (.+)$", e)
        if not m:
            continue
        src = os.path.expanduser(m.group(1).strip())
        if not os.path.exists(src):
            continue
        try:
            dst_dir = os.path.join(os.getcwd(), "downloads")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            extra.append(f"[download] copied into workspace: {dst}")
        except Exception:
            pass
    return extra


def process_browser_response(data, action_desc, session_id, tab_id,
                             include_vision=True, visual="overlay", worker=None,
                             compare=False):
    if data.get("status") == "error":
        return f"Browser Error during {action_desc}: {data.get('msg')}"

    # Plain-text results (get_text / read_text) have no screenshot payload.
    if "clean_b64" not in data and "text" in data:
        return f"--- {action_desc} ---\nExtracted text:\n{data['text'][:8000]}"

    output_dir = os.path.expanduser(f"~/.aeon/temp/browser_output_{session_id}_{tab_id}")
    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "clean.jpg")
    overlay_path = os.path.join(output_dir, "overlay.jpg")
    prev_clean_path = os.path.join(output_dir, "prev_clean.jpg")
    prev_overlay_path = os.path.join(output_dir, "prev_overlay.jpg")

    # For a before/after comparison, preserve LAST turn's frames (still on disk)
    # as prev_* BEFORE we overwrite them with this turn's screenshots.
    if compare:
        for src, dst in ((clean_path, prev_clean_path), (overlay_path, prev_overlay_path)):
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

    clean_bytes = base64.b64decode(data["clean_b64"])
    with open(clean_path, "wb") as f:
        f.write(clean_bytes)
    have_overlay = bool(data.get("overlay_b64"))
    if have_overlay:
        with open(overlay_path, "wb") as f:
            f.write(base64.b64decode(data["overlay_b64"]))

    # Human-facing terminal preview of the real (clean) render.
    _print_image_to_terminal(clean_bytes, target_width=80)

    elements_str = _format_elements(data.get("elements", []))
    scroll_str = _format_scroll(data.get("page_state"))
    page_title = data.get("title", "Unknown")
    page_url = data.get("url", "Unknown")
    open_tabs = data.get("open_tabs", [])
    open_tabs_str = ", ".join(open_tabs) if open_tabs else tab_id

    # --- Perception: hand the ACTUAL rendered page to the deciding model ---
    # No separate caption model: the multimodal primary looks at the screenshot
    # itself on its next turn (worker.set_visual_context), exactly as a human sees
    # the page, with the structured element list as id grounding. `visual` selects
    # which frame(s) the model gets:
    #   overlay (default) -> the render + numbered [id] marks (best action grounding)
    #   clean             -> the pure render, no marks (closest to a bare human view)
    #   both              -> clean first, then the numbered overlay
    attached = []
    if include_vision and worker is not None:
        if visual in ("clean", "both"):
            attached.append(clean_path)
        if visual in ("overlay", "both") and have_overlay:
            attached.append(overlay_path)
        if not attached:  # e.g. overlay requested but server didn't render it
            attached.append(clean_path)
        # Before/after: prepend last turn's matching frame so the model can diff
        # what this action changed (the "video" case, done as two still frames).
        prev_for_compare = prev_overlay_path if (have_overlay and visual != "clean") else prev_clean_path
        compared = compare and os.path.exists(prev_for_compare)
        if compared:
            attached = [prev_for_compare] + attached
        try:
            worker.set_visual_context(attached)
        except Exception:
            attached = []
            compared = False
    else:
        compared = False

    if attached:
        marks = " (with numbered [id] marks)" if any(p in (overlay_path, prev_overlay_path) for p in attached) else ""
        if compared:
            vision_note = (f"TWO screenshots{marks} are attached to your NEXT turn: the FIRST is the page "
                           f"BEFORE this action, the SECOND is AFTER. Compare them to see exactly what "
                           f"changed, then act by [id].")
        else:
            vision_note = (f"A screenshot of this page{marks} is attached to your NEXT turn — look at it "
                           f"directly to see the page as a human would, then act by [id].")
    elif not include_vision:
        vision_note = ("(screenshot NOT attached this step for speed — you are acting on the element "
                       "list alone; set include_vision=true or call browser_read to see the page.)")
    else:
        vision_note = ("(no multimodal context available to attach the screenshot; acting on the "
                       "element list. Screenshots saved to disk below.)")

    # Dialogs auto-handled and files downloaded since the last step are reported
    # here so the agent always knows they happened. Downloaded files are also
    # copied into the workspace ./downloads so the agent can use them directly.
    events = list(data.get("events") or [])
    events += _relocate_downloads(events)
    events_str = ("\n=== EVENTS ===\n" + "\n".join(f"- {e}" for e in events) + "\n") if events else ""

    shots = f"clean={clean_path}" + (f" | numbered={overlay_path}" if have_overlay else "")
    return (
        f"--- BROWSER: {action_desc} (tab '{tab_id}') ---\n"
        f"URL: {page_url}\n"
        f"Title: {page_title}\n"
        f"Open tabs: [{open_tabs_str}]\n"
        f"{scroll_str}\n"
        f"{events_str}\n"
        f"=== INTERACTIVE ELEMENTS (act on these by [id]) ===\n"
        f"{elements_str}\n\n"
        f"=== PAGE VIEW ===\n{vision_note}\n"
        f"Screenshots on disk: {shots}"
    )


def _post(endpoint, payload, action_desc, tab_id, timeout=90,
          include_vision=True, visual="overlay", worker=None, compare=False):
    try:
        ensure_browser_running()
        # The server only needs to draw+shoot the numbered overlay when we will
        # actually attach the overlay frame. Skipping it otherwise is a free
        # latency cut (one screenshot + two page evals).
        want_overlay = include_vision and visual in ("overlay", "both")
        payload = {**payload, "overlay": want_overlay, "profile": _profile_for(worker)}
        resp = requests.post(f"{BROWSER_API_URL}/{endpoint}", json=payload, timeout=timeout)
        if resp.status_code != 200:
            # The server returns a helpful 'detail' for 4xx (e.g. expected_text mismatch).
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return f"Browser action failed ({action_desc}): HTTP {resp.status_code}: {detail}"
        return process_browser_response(resp.json(), action_desc, payload.get("session_id"),
                                        tab_id, include_vision=include_vision,
                                        visual=visual, worker=worker, compare=compare)
    except Exception as e:
        return f"Error during {action_desc}: {type(e).__name__}: {e}"


class BrowserNavigateTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_navigate", description=TOOL_DESC_BROWSER_NAVIGATE)
        self.worker = worker

    def execute(self, url: str, tab_id: str = "default", include_vision: bool = True,
                visual: str = "overlay", **kwargs) -> str:
        if not url:
            return "Error: 'url' is required."
        return _post("navigate", {"session_id": _session_id(), "tab_id": tab_id, "url": url},
                     f"Navigated to {url}", tab_id, include_vision=include_vision,
                     visual=visual, worker=self.worker)


class BrowserInteractTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_interact", description=TOOL_DESC_BROWSER_INTERACT)
        self.worker = worker

    def execute(self, action: str, element_id: int = None, text: str = None,
                expected_text: str = None, tab_id: str = "default", key: str = None,
                value: str = None, file_path: str = None, amount: int = None,
                direction: str = None, to_element_id: int = None,
                clear_first: bool = True, then_enter: bool = False,
                duration: int = 2000, include_vision: bool = True,
                visual: str = "overlay", compare: bool = False, **kwargs) -> str:
        if not action:
            return "Error: 'action' is required."
        # Friendly aliases so common phrasings just work.
        alias = {"scroll_down": ("scroll", "down"), "scroll_up": ("scroll", "up"),
                 "scroll_to_bottom": ("scroll", "bottom"), "scroll_to_top": ("scroll", "top"),
                 "scroll_left": ("scroll", "left"), "scroll_right": ("scroll", "right"),
                 "scroll_to_left": ("scroll", "leftmost"), "scroll_to_right": ("scroll", "rightmost"),
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
        return _post("interact", payload, desc, tab_id, include_vision=include_vision,
                     visual=visual, worker=self.worker, compare=compare)


class BrowserReadTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="browser_read",
            description=(
                "Re-observe the current page WITHOUT acting: returns a fresh indexed element list and "
                "scroll state, and attaches the current page screenshot to your next turn so you SEE the "
                "page again. Use it after a page updates on its own, when element ids feel stale, or to "
                "re-orient (take a fresh look) before choosing the next action.\n"
                "Schema:\n  tab_id (str, optional, default='default'): the tab to read.\n"
                "  include_vision (bool, optional, default true): set false to skip attaching the "
                "screenshot for a faster, element-list-only read when you don't need to see the page.\n"
                "  visual (str, optional, default 'overlay'): 'overlay' = render + numbered [id] marks; "
                "'clean' = pure render, no marks; 'both'.\n"
                "Example: {\"tool_name\": \"browser_read\", \"parameters\": {\"tab_id\": \"gmail\"}}"
            ),
        )
        self.worker = worker

    def execute(self, tab_id: str = "default", include_vision: bool = True,
                visual: str = "overlay", **kwargs) -> str:
        return _post("observe", {"session_id": _session_id(), "tab_id": tab_id}, "Read page", tab_id,
                     include_vision=include_vision, visual=visual, worker=self.worker)


class BrowserCloseTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_close_tab", description=TOOL_DESC_BROWSER_CLOSE_TAB)
        self.worker = worker

    def execute(self, tab_id: str) -> str:
        try:
            ensure_browser_running()
            resp = requests.post(f"{BROWSER_API_URL}/close_tab",
                                 json={"session_id": _session_id(), "tab_id": tab_id,
                                       "profile": _profile_for(self.worker)}, timeout=15)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            data = resp.json()
            remaining = data.get("remaining_tabs", 0)
            if remaining == 0:
                rem_browser = _manage_browser_registry('unregister')
                if rem_browser == 0:
                    subprocess.run(['docker', 'rm', '-f', 'aeon_browser'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Closed tab '{tab_id}'. No tabs left; released the browser resource."
            return f"Closed tab '{tab_id}'. {remaining} tab(s) still open."
        except Exception as e:
            return self.format_error_message(e, f"closing tab {tab_id}")


class BrowserSwitchTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_switch_tab", description=TOOL_DESC_BROWSER_SWITCH_TAB)
        self.worker = worker

    def execute(self, tab_id: str = "default", include_vision: bool = True,
                visual: str = "overlay", **kwargs) -> str:
        return _post("switch_tab", {"session_id": _session_id(), "tab_id": tab_id},
                     f"Switched to tab '{tab_id}'", tab_id,
                     include_vision=include_vision, visual=visual, worker=self.worker)
