import os
import re
import time
import base64
import requests
import subprocess
import json
import fcntl
import hashlib
import io
import shutil
import stat
import uuid
from datetime import datetime
from PIL import Image
from .base import BaseTool
from ..core.prompts import (
    TOOL_DESC_BROWSER_NAVIGATE,
    TOOL_DESC_BROWSER_INTERACT,
    TOOL_DESC_BROWSER_CLOSE_TAB,
    TOOL_DESC_BROWSER_SWITCH_TAB,
    TOOL_DESC_BROWSER_CAPTURE_MEDIA,
)
from ..core.paths import resolve_output_dir
from ..core.model_catalog import VISION_MODEL_NAME
from ..services.browser.browser_util import read_auth_token

BROWSER_API_URL = "http://localhost:8030"
BROWSER_API_VERSION = "human_v6"

# Cap the structured element list so a huge page can't flood context. The agent
# can scroll to reveal more; function is primary but context still must survive.
MAX_ELEMENT_LINES = 160
MAX_ELEMENTS_CHARS = 14000
MAX_VISIBLE_TEXT_CHARS = 5000
MAX_BROWSER_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_TARGET_CROPS = 2
TARGET_CROP_SCALE = 2
TARGET_CROP_MAX_SOURCE_DIM = 960  # 2x stays within LLMClient.VISION_MAX_DIM=1920

_BROWSER_ACTION_ALIASES = {
    "scroll_down": ("scroll", "down"),
    "scroll_up": ("scroll", "up"),
    "scroll_to_bottom": ("scroll", "bottom"),
    "scroll_to_top": ("scroll", "top"),
    "scroll_left": ("scroll", "left"),
    "scroll_right": ("scroll", "right"),
    "scroll_to_left": ("scroll", "leftmost"),
    "scroll_to_right": ("scroll", "rightmost"),
    "enter": ("press_key", None),
    "wait": ("wait_for", None),
    "select": ("select_option", None),
    "choose": ("select_option", None),
    "pick": ("select_option", None),
}
_BROWSER_ACTIONS = frozenset({
    "click", "double_click", "right_click", "hover", "type", "press_key",
    "scroll", "select_option", "check", "uncheck", "clear", "drag",
    "press_and_hold", "upload_file", "go_back", "go_forward", "reload",
    "wait_for", "get_text", "read_text", "click_at", "double_click_at",
    "right_click_at", "hover_at", "type_at", "drag_at", "press_and_hold_at",
})


def _normalize_browser_action(action, text, duration, direction, key, value):
    """Canonicalize legacy/human action names through one tested contract."""
    original = str(action or "").strip().lower()
    canonical, implied_direction = _BROWSER_ACTION_ALIASES.get(
        original, (original, None)
    )
    if canonical == "scroll" and direction is None:
        direction = implied_direction
    if original == "enter":
        key = key or "Enter"
    if canonical == "select_option" and value is None and text is not None:
        value = text
    # Backward compatibility for the old documented wait action: wait(text="5")
    # meant five seconds.  Canonical wait_for(text="Ready") still waits for text.
    if original == "wait" and text is not None:
        try:
            seconds = float(str(text).strip())
        except (TypeError, ValueError):
            pass
        else:
            duration = max(0, min(int(seconds * 1000), 120000))
            text = None
    return canonical, text, duration, direction, key, value


def _stage_browser_upload(file_path):
    """Copy a host workspace file into the browser's private mounted volume."""
    if not file_path or not str(file_path).strip():
        raise ValueError("upload_file requires 'file_path'.")
    source = os.path.realpath(os.path.expanduser(str(file_path)))
    try:
        info = os.stat(source, follow_symlinks=True)
    except OSError as exc:
        raise ValueError(f"upload source is unavailable: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("upload source must be a regular file.")
    if info.st_size > MAX_BROWSER_UPLOAD_BYTES:
        raise ValueError(
            f"upload source exceeds the {MAX_BROWSER_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )
    host_dir = os.path.join(_host_download_dir(), "..", "uploads")
    host_dir = os.path.realpath(host_dir)
    os.makedirs(host_dir, mode=0o700, exist_ok=True)
    os.chmod(host_dir, 0o700)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(source)) or "upload"
    staged_name = f"{uuid.uuid4().hex}-{safe_name}"
    staged_host = os.path.join(host_dir, staged_name)
    shutil.copy2(source, staged_host)
    os.chmod(staged_host, 0o600)
    return staged_host, f"/profiles/uploads/{staged_name}"

# --- Ground-truth diagnostics -------------------------------------------------
# The agent's own "PREVIOUS RESULT SUMMARY" is model narration and, on a weak
# model, cannot be trusted to say what the page really was. This log records the
# RAW browser facts per action — real URL, whether the URL actually changed, the
# actual element list, and whether the screenshot was a valid frame or blank — so
# a stuck run can be diagnosed from ground truth. Location is fixed and printed to
# the terminal on first use so it's easy to find.
BROWSER_DIAG_LOG = os.path.expanduser("~/.aeon/logs/browser_diag.log")
_diag_last_url = {}       # (session, tab) -> last URL, to flag actions that didn't navigate
_diag_run_started = False  # so we stamp a "NEW RUN" banner once per process
_last_page_sig = {}       # (session, tab) -> signature of the last observed page (model-facing no-op detector)


def _browser_token_path():
    return os.environ.get(
        "AEON_BROWSER_TOKEN_FILE",
        os.path.join(os.environ.get("AEON_HOME") or os.path.expanduser("~/.aeon"),
                     "browser_api_token"),
    )


def browser_auth_headers():
    """Login header for the localhost browser controller.

    Read on every request so credential rotation does not require restarting
    Aeon. ``read_auth_token`` rejects missing or over-permissive secret files.
    """
    token = read_auth_token(_browser_token_path())
    return {"Authorization": f"Bearer {token}"}


def _worker_uses_qwen38_vision(worker):
    """True only when browser screenshots would reach the approved Qwen3.8 ID."""
    return (getattr(getattr(worker, "llm_client", None), "api_model", None)
            == VISION_MODEL_NAME)


def _page_signature(data):
    """Compact, comparable signature of what the agent can actually act on: the URL
    plus the ordered (id, role, name/value, states) of every interactive element.
    Two consecutive turns with the SAME signature are the same actionable page — the
    action between them accomplished nothing the agent can act on. This is the fact
    that stops a weak model from re-clicking a dead button, so — unlike the diag log
    — it must be surfaced INTO the result the model reads."""
    els = data.get("elements") or []
    el_sig = tuple(
        (e.get("id"), e.get("role"),
         (e.get("name") or e.get("value") or "").strip(),
         tuple(e.get("states") or []))
        for e in els
    )
    # Ignore changing numbers (clocks, unread counts, ad timers) so they do not
    # masquerade as task progress, while still noticing real non-interactive text
    # changes such as an error banner, article transition, or modal message.
    visible = re.sub(r"\d+", "#", str(data.get("visible_text") or ""))
    visible = re.sub(r"\s+", " ", visible).strip()[:MAX_VISIBLE_TEXT_CHARS]
    return (data.get("url", ""), el_sig, visible)


def _format_validation(validation):
    """Turn the server's validation scrape into a short, actionable block for the
    agent — this is what tells a weak model WHY a Submit click did nothing (a
    required field is empty / a dropdown is unset) instead of leaving it to read
    small red error text off a screenshot. Returns '' when the form looks clean."""
    if not validation:
        return ""
    invalid = validation.get("invalid") or []
    alerts = validation.get("alerts") or []
    if not invalid and not alerts:
        return ""
    lines = ["=== FORM VALIDATION (fix these before the form will submit) ==="]
    for item in invalid[:20]:
        lines.append(f"- {item.get('label', '?')}: {item.get('reason', 'invalid')}")
    for a in alerts[:8]:
        lines.append(f"- error message: {a}")
    return "\n".join(lines)


def _screenshot_health(clean_bytes):
    """Return (summary, is_blank). Tells whether the model was actually SEEING the
    page or handed a genuinely blank/degraded frame — the difference between "this
    page is just sparse" and "our vision pipeline broke and the model would
    confabulate".

    Blankness is the fraction of pixels that DEPART from the background luminance
    (real content/text), not global std. This matters: a legible page like
    example.com is mostly uniform background with one small block of text, so its
    global std is low — the old test downsampled to 32x32 (smearing the text away)
    and mislabeled such perfectly-readable pages as blank, which would then tell
    the model to distrust a screenshot it could actually read. Text survives as a
    small-but-clear content fraction (~1% on example.com) while a truly blank
    frame has ~0. Require BOTH almost-no-content AND a tiny luminance range so
    only a genuinely uniform frame trips it.
    """
    if not clean_bytes:
        return "MISSING (no screenshot bytes in response)", True
    size = len(clean_bytes)
    try:
        img = Image.open(io.BytesIO(clean_bytes)).convert("L")
        w, h = img.size
        # 256x256: bounds cost, drops JPEG noise, but keeps real text as content.
        px = sorted(img.resize((256, 256)).getdata())
        n = len(px)
        bg = px[n // 2]                                   # median = background
        content = sum(1 for p in px if abs(p - bg) > 24) / n
        lum_range = px[-1] - px[0]
        # example.com scores content~0.010 / range~136; a uniform failed paint
        # scores ~0.000 / ~0. The gap is ~20x, so this has wide margin either way.
        is_blank = content < 0.0005 and lum_range < 24
        flag = "  <-- LIKELY BLANK/UNIFORM (model may be flying blind)" if is_blank else ""
        return (f"bytes={size} dims={w}x{h} bg_lum={bg} content={content * 100:.2f}% "
                f"range={lum_range}{flag}", is_blank)
    except Exception as e:
        tiny = size < 3000
        note = "  <-- suspiciously small (possibly blank)" if tiny else ""
        return f"bytes={size} (decode failed: {e}){note}", tiny


def _valid_viewport_rect(rect):
    """Normalize a browser viewport rect, or return None for malformed geometry."""
    if not isinstance(rect, dict):
        return None
    try:
        x, y = float(rect.get("x", 0)), float(rect.get("y", 0))
        w, h = float(rect.get("w", 0)), float(rect.get("h", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _rect_iou(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _target_crop_regions(data, focus_element_id=None, focus_point=None,
                         focus_text=""):
    """Rank grounded viewport regions that merit a lossless enlarged crop.

    Sources, in priority order: the control/coordinate just acted on; explicit
    CAPTCHA/verification and error regions reported by the browser service;
    invalid form controls; dense tables and diagrams.  This is deterministic and
    uses browser geometry only—there is no extra caption-model call.
    """
    candidates = []

    def add(priority, kind, label, rect):
        normalized = _valid_viewport_rect(rect)
        if normalized is None:
            return
        for existing in candidates:
            if _rect_iou(existing["rect"], normalized) >= 0.72:
                return
        candidates.append({
            "priority": int(priority),
            "kind": str(kind or "target")[:32],
            "label": re.sub(r"\s+", " ", str(label or "")).strip()[:120],
            "rect": normalized,
        })

    # The service captures this rectangle before acting, after resolving stale ids
    # and scrolling the real target into view.  Prefer it to looking the numeric id
    # up in the post-action DOM, where navigation/re-rendering may have reused that
    # id for a completely different control.
    action_focus = data.get("action_focus") or {}
    if isinstance(action_focus, dict):
        source_url = str(action_focus.get("source_url") or "")
        current_url = str(data.get("url") or "")
        if not source_url or source_url == current_url:
            add(0, "target", action_focus.get("label") or "intended control",
                action_focus.get("rect"))

    elements = [e for e in (data.get("elements") or [])
                if isinstance(e, dict) and e.get("inViewport")]
    if focus_element_id is not None and not any(c["priority"] == 0 for c in candidates):
        for element in elements:
            if element.get("id") == focus_element_id:
                add(0, "target", f"element [{focus_element_id}]", element.get("rect"))
                break
    if focus_text and not any(c["priority"] == 0 for c in candidates):
        needle = re.sub(r"\s+", " ", str(focus_text)).strip().casefold()
        if needle:
            matches = []
            for element in elements:
                hay = re.sub(
                    r"\s+", " ", str(element.get("name") or element.get("value") or "")
                ).strip().casefold()
                if hay and (needle in hay or hay in needle):
                    matches.append(element)
            if len(matches) == 1:
                add(0, "target", "intended control", matches[0].get("rect"))
    if focus_point and not any(c["priority"] == 0 for c in candidates):
        try:
            px, py = float(focus_point[0]), float(focus_point[1])
            add(0, "target", "coordinate target", {"x": px - 2, "y": py - 2,
                                                     "w": 4, "h": 4})
        except (TypeError, ValueError, IndexError):
            pass

    region_priority = {"verification": 1, "captcha": 1, "error": 2,
                       "table": 3, "diagram": 4}
    for region in data.get("visual_regions") or []:
        if not isinstance(region, dict):
            continue
        kind = str(region.get("kind") or "visual").lower()
        if kind not in region_priority:
            continue
        add(region_priority[kind], kind, region.get("label") or kind,
            region.get("rect"))

    # Native/ARIA validation identifies the bad field even when an error banner
    # itself has no useful rectangle.
    validation_labels = [
        str(item.get("label") or "").strip().casefold()
        for item in ((data.get("validation") or {}).get("invalid") or [])
        if isinstance(item, dict) and item.get("label")
    ]
    for label in validation_labels:
        for element in elements:
            name = str(element.get("name") or "").strip().casefold()
            if name and (label in name or name in label):
                add(2, "error", "invalid form control", element.get("rect"))
                break

    # Older browser-service responses do not have visual_regions. Preserve a
    # useful fallback by unioning a dense cluster of row/cell geometry.
    dense = [e for e in elements if str(e.get("role") or "").lower()
             in {"row", "cell", "gridcell"} and _valid_viewport_rect(e.get("rect"))]
    if len(dense) >= 6 and not any(c["kind"] == "table" for c in candidates):
        rects = [_valid_viewport_rect(e["rect"]) for e in dense]
        x1, y1 = min(r["x"] for r in rects), min(r["y"] for r in rects)
        x2 = max(r["x"] + r["w"] for r in rects)
        y2 = max(r["y"] + r["h"] for r in rects)
        add(3, "table", "dense table/grid", {"x": x1, "y": y1,
                                               "w": x2 - x1, "h": y2 - y1})

    candidates.sort(key=lambda item: (item["priority"],
                                      item["rect"]["w"] * item["rect"]["h"]))
    return candidates


def _crop_box_for_region(rect, image_size, kind="target"):
    """Expand a viewport region to a useful context window, clamped to the image."""
    image_w, image_h = image_size
    cx = rect["x"] + rect["w"] / 2
    cy = rect["y"] + rect["h"] / 2
    if kind in {"table", "diagram"}:
        desired_w = max(640.0, rect["w"] + 120.0)
        desired_h = max(420.0, rect["h"] + 120.0)
    else:
        desired_w = max(480.0, rect["w"] * 3.0 + 160.0)
        desired_h = max(320.0, rect["h"] * 5.0 + 140.0)
    desired_w = min(float(image_w), float(TARGET_CROP_MAX_SOURCE_DIM), desired_w)
    desired_h = min(float(image_h), float(TARGET_CROP_MAX_SOURCE_DIM), desired_h)
    left = max(0.0, min(float(image_w) - desired_w, cx - desired_w / 2))
    top = max(0.0, min(float(image_h) - desired_h, cy - desired_h / 2))
    return (int(round(left)), int(round(top)),
            int(round(left + desired_w)), int(round(top + desired_h)))


def _write_target_crops(clean_path, output_dir, data, focus_element_id=None,
                        focus_point=None, focus_text="", limit=MAX_TARGET_CROPS):
    """Write up to ``limit`` atomic, lossless 2x PNG crops; return (path,label)."""
    try:
        regions = _target_crop_regions(
            data, focus_element_id=focus_element_id,
            focus_point=focus_point, focus_text=focus_text)
        if not regions:
            return []
        results = []
        with Image.open(clean_path) as source:
            source.load()
            image_size = source.size
            for region in regions:
                if len(results) >= max(0, int(limit)):
                    break
                box = _crop_box_for_region(region["rect"], image_size, region["kind"])
                crop_w, crop_h = box[2] - box[0], box[3] - box[1]
                # Enlarging almost the entire 1920px frame adds no visual detail;
                # the full screenshot already supplies that context.
                if crop_w * crop_h >= image_size[0] * image_size[1] * 0.82:
                    continue
                crop = source.crop(box)
                resampling = getattr(Image, "Resampling", Image)
                crop = crop.resize(
                    (crop_w * TARGET_CROP_SCALE, crop_h * TARGET_CROP_SCALE),
                    resampling.LANCZOS,
                )
                path = os.path.join(
                    output_dir, f"target_{len(results) + 1}_{region['kind']}_2x.png")
                tmp = path + ".tmp"
                crop.save(tmp, format="PNG")
                os.replace(tmp, path)
                label = (f"lossless 2x {region['kind']} crop"
                         + (f" ({region['label']})" if region["label"] else ""))
                results.append((path, label))
        return results
    except Exception:
        # A crop is an enhancement; the full screenshot remains authoritative and
        # must never be lost because one malformed page rectangle reached us.
        return []


def _compact_elements(elements, limit=80):
    """One line per element: [id] role 'name' (states) — enough to see what the
    agent was actually clicking (e.g. whether [7] was really a 'Continue' button)."""
    if not elements:
        return "   (none detected)"
    lines = []
    for e in elements[:limit]:
        name = (e.get("name") or e.get("value") or "").replace("\n", " ").strip()
        if len(name) > 70:
            name = name[:70] + "…"
        states = e.get("states") or []
        st = f" ({', '.join(states)})" if states else ""
        vp = "" if e.get("inViewport") else " [off-screen]"
        lines.append(f"   [{e.get('id')}] {e.get('role', '?')} '{name}'{st}{vp}")
    extra = len(elements) - limit
    if extra > 0:
        lines.append(f"   … (+{extra} more)")
    return "\n".join(lines)


def _log_browser_diag(data, action_desc, session_id, tab_id, clean_bytes=None):
    """Append a raw, model-independent record of one browser action to
    BROWSER_DIAG_LOG, and print a one-line summary to the terminal. Best-effort:
    diagnostics must never break the browser tool."""
    global _diag_run_started
    try:
        status = data.get("status", "ok")
        if status == "error":
            body = f"status    : ERROR — {data.get('msg')}"
            term = f"[browser-diag] {action_desc}: ERROR — {str(data.get('msg'))[:120]}"
        else:
            url = data.get("url", "Unknown")
            key = (session_id, tab_id)
            prev = _diag_last_url.get(key)
            if prev is None:
                changed = "first observation"
            elif prev == url:
                changed = "UNCHANGED from previous call  <-- this action did NOT navigate"
            else:
                changed = "changed"
            _diag_last_url[key] = url
            els = data.get("elements") or []
            if clean_bytes is None and data.get("clean_b64"):
                try:
                    clean_bytes = base64.b64decode(data["clean_b64"])
                except Exception:
                    clean_bytes = None
            shot, is_blank = _screenshot_health(clean_bytes)
            events = data.get("events") or []
            ev = "\n".join(f"   - {e}" for e in events) if events else "   (none)"
            validation = data.get("validation") or {}
            vinvalid = validation.get("invalid") or []
            valerts = validation.get("alerts") or []
            if vinvalid or valerts:
                vlines = [f"   - {i.get('label', '?')}: {i.get('reason', 'invalid')}" for i in vinvalid]
                vlines += [f"   - error: {a}" for a in valerts]
                vstr = "\n".join(vlines)
            else:
                vstr = "   (form looks valid / none)"
            identity = data.get("identity")
            body = (
                f"status    : {status}\n"
                f"URL       : {url}  ({changed})\n"
                f"title     : {data.get('title', '')}\n"
                f"identity  : {identity or '(not signed in / not detected)'}\n"
                f"screenshot: {shot}\n"
                f"validation:\n{vstr}\n"
                f"events    :\n{ev}\n"
                f"elements  ({len(els)} total):\n{_compact_elements(els)}"
            )
            url_flag = "URL UNCHANGED" if (prev is not None and prev == url) else "url ok"
            shot_flag = "SCREENSHOT BLANK" if is_blank else "screenshot ok"
            vflag = f" | {len(vinvalid)} VALIDATION ISSUE(S)" if (vinvalid or valerts) else ""
            idflag = f" | as {identity}" if identity else ""
            term = f"[browser-diag] {action_desc}: {url_flag} | {shot_flag} | {len(els)} elements{vflag}{idflag}"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        header = f"\n{'=' * 26} BROWSER DIAG {ts} {'=' * 26}\n"
        entry = header + f"action    : {action_desc}\ntab       : {tab_id}\n" + body + "\n"

        os.makedirs(os.path.dirname(BROWSER_DIAG_LOG), exist_ok=True)
        if not _diag_run_started:
            entry = f"\n\n{'#' * 20} NEW AEON RUN @ {ts} {'#' * 20}\n" + entry
            _diag_run_started = True
            print(f"[browser-diag] logging raw browser facts to: {BROWSER_DIAG_LOG}")
        with open(BROWSER_DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
        print(term)
    except Exception:
        pass  # never let diagnostics break a browser action


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
        response = requests.get(
            f"{BROWSER_API_URL}/health", headers=browser_auth_headers(), timeout=2
        )
        if response.status_code != 200:
            return False
        body = response.json()
        # Do not treat the legacy unauthenticated server as healthy merely
        # because it ignores our Authorization header and returns HTTP 200.
        return (
            body.get("status") == "ok"
            and body.get("auth_required") is True
            and body.get("api_version") == BROWSER_API_VERSION
        )
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
        # Bound the start: `docker run -d` returns fast, but a stuck docker
        # daemon must not hang the agent loop. Capture output so a failed start
        # yields a diagnostic instead of a bare CalledProcessError.
        try:
            subprocess.run(["bash", script_path], check=True,
                           capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out (180s) starting the browser service (docker may be stuck).")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start the browser service: {(e.stderr or e.stdout or '').strip()[:400]}")
    return True


def _session_id():
    return str(os.getpid())


def _resolve_tab(worker, tab_id):
    """Resolve the tab to act on, defaulting to the LAST tab this agent used rather
    than the literal 'default'. The #1 browser footgun was: the model navigates to
    tab 'email_task' but a later interact omits tab_id -> it defaulted to 'default'
    (a tab that never existed) -> 'HTTP 404: Tab default not found' -> the agent
    blamed itself and looped. Now an omitted/'default' tab_id follows the last tab
    the agent actually navigated/acted in."""
    last = getattr(worker, "_last_browser_tab", None) if worker else None
    if not tab_id or tab_id == "default":
        return last or "default"
    return tab_id


def _remember_tab(worker, tab_id):
    """Record the tab this agent is currently working in, for _resolve_tab."""
    if worker is not None and tab_id:
        try:
            worker._last_browser_tab = tab_id
        except Exception:
            pass


def _profile_for(worker):
    """The browser profile (isolation unit) for this agent. The principal uses
    'default' (shared, persistent — logins survive); each sub-agent sets its own
    on the worker so it browses as an independent identity."""
    raw = str(getattr(worker, "browser_profile", "default") or "default")
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in raw)
    return safe.strip("-.")[:64] or "default"


class _BrowserTabLock:
    """Cross-process lock for one persistent browser profile/session/tab.

    Aeon agents may share the browser service. Serializing only the tab being
    acted on prevents a read/action race without unnecessarily blocking other
    profiles or tabs.
    """

    def __init__(self, profile, session_id, tab_id):
        identity = f"{profile}\0{session_id}\0{tab_id}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        root = os.path.join(
            os.environ.get("AEON_HOME") or os.path.expanduser("~/.aeon"),
            "browser_locks",
        )
        os.makedirs(root, mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        self.path = os.path.join(root, f"{digest}.lock")
        self.fd = None

    def __enter__(self):
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.fd = os.open(self.path, flags, 0o600)
        os.fchmod(self.fd, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


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


def _host_download_dir():
    """Host path of the container's DOWNLOAD_DIR (/profiles/downloads). The browser
    profile volume ${AEON_HOME:-~/.aeon}/browser_profiles is mounted at /profiles
    (see start_browser.sh), so a file the service wrote there is visible here."""
    aeon_home = os.environ.get("AEON_HOME") or os.path.expanduser("~/.aeon")
    return os.path.join(aeon_home, "browser_profiles", "downloads")


def _format_media_list(data):
    """Render the enumerated page media so the model can pick one by id."""
    media = data.get("media") or []
    if not media:
        return ("No images or videos were detected on this page. If you expected some, scroll the "
                "media into view or wait for it to load, then call browser_capture_media again.")
    lines = [f"Media found on {data.get('url', '')}. To SAVE one, call browser_capture_media again "
             f"with its media_id AND an output_dir:"]
    for m in media:
        alt = f' — "{m["alt"]}"' if m.get("alt") else ""
        src = m.get("src") or ""
        src_hint = (" src=" + (src if len(src) <= 80 else src[:77] + "…")) if src else ""
        loc = "in view" if m.get("inView") else "off-screen"
        lines.append(f"  [media_id {m.get('id')}] {m.get('tag')} {m.get('w')}x{m.get('h')} "
                     f"({m.get('dw')}x{m.get('dh')}px on screen, {loc}){alt}{src_hint}")
    return "\n".join(lines)


def process_browser_response(data, action_desc, session_id, tab_id,
                             include_vision=True, visual="overlay", worker=None,
                             compare=False, focus_element_id=None,
                             focus_point=None, focus_text=""):
    if data.get("status") == "error":
        _log_browser_diag(data, action_desc, session_id, tab_id)
        return f"Browser Error during {action_desc}: {data.get('msg')}"

    # Plain-text results (get_text / read_text) have no screenshot payload.
    if "clean_b64" not in data and "text" in data:
        _log_browser_diag(data, action_desc, session_id, tab_id)
        return f"--- {action_desc} ---\nExtracted text:\n{data['text'][:20000]}"

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
    # Ground-truth diagnostics from the raw response (real URL, URL-change, element
    # list, screenshot health) — independent of the model's later summary.
    _log_browser_diag(data, action_desc, session_id, tab_id, clean_bytes=clean_bytes)
    # Surface screenshot health to the MODEL, not just the diag log. A blank/uniform
    # frame means the render failed and the model is flying blind; without telling it,
    # the model confidently "reads" content off an empty image (observed on
    # example.com: a blank frame, yet the agent reported the h1 from priors and
    # declared success). Ground it in the truth instead.
    try:
        _, screenshot_is_blank = _screenshot_health(clean_bytes)
    except Exception:
        screenshot_is_blank = False
    have_overlay = bool(data.get("overlay_b64"))
    if have_overlay:
        with open(overlay_path, "wb") as f:
            f.write(base64.b64decode(data["overlay_b64"]))

    # --- Model-facing NO-OP detection ---
    # Compare this page to the last one observed on the same tab. The deciding model
    # only ever holds the CURRENT page in context (last turn's is gone), so it cannot
    # tell on its own that an action changed nothing — and an unchanged page that
    # still shows the button it just clicked makes re-clicking that button the most
    # natural next move. Telling it plainly "this did nothing, the fix is elsewhere"
    # removes the pull to repeat at the source, before the loop guard ever has to.
    key = (session_id, tab_id)
    sig = _page_signature(data)
    prev_sig = _last_page_sig.get(key)
    no_change = prev_sig is not None and prev_sig == sig
    _last_page_sig[key] = sig
    # A read is meant only to re-observe; a click/type/navigate is meant to change
    # something, so an unchanged page there is a wasted action worth flagging hard.
    lowered_action = action_desc.strip().lower()
    is_passive_read = lowered_action.startswith("read page") or lowered_action.startswith("find ")
    if no_change and not is_passive_read:
        no_change_block = (
            "\n⚠ NO CHANGE: the URL and EVERY interactive element are identical to before this "
            "action — it had NO EFFECT on the page. Do NOT repeat it (or minor variants of it); an "
            "inert action cannot change the result by being retried. The real cause is elsewhere — "
            "typically ONE of: a required field is empty/invalid (check FORM VALIDATION below), the "
            "target is disabled or not the real control, the control you need is a DIFFERENT element, "
            "or you must scroll/switch focus/tab first. Pick a DIFFERENT element or a different "
            "approach this turn.\n"
        )
    elif no_change:
        no_change_block = "\n(No change since your last observation — the page has not updated.)\n"
    else:
        no_change_block = ""

    elements_str = _format_elements(data.get("elements", []))
    visible_text = str(data.get("visible_text") or "").strip()
    if len(visible_text) > MAX_VISIBLE_TEXT_CHARS:
        visible_text = visible_text[:MAX_VISIBLE_TEXT_CHARS] + "\n… [visible text truncated]"
    visible_text_block = (
        "\n=== VISIBLE PAGE TEXT (grounded DOM text) ===\n" + visible_text + "\n"
        if visible_text else ""
    )
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
    attachment_labels = []
    target_crops = []
    worker_vision_model = getattr(getattr(worker, "llm_client", None), "api_model", None)
    qwen38_vision_ready = _worker_uses_qwen38_vision(worker)
    if include_vision and worker is not None and qwen38_vision_ready:
        if visual in ("clean", "both"):
            attached.append(clean_path)
            attachment_labels.append("current full clean screenshot")
        if visual in ("overlay", "both") and have_overlay:
            attached.append(overlay_path)
            attachment_labels.append("current full screenshot with numbered [id] marks")
        if not attached:  # e.g. overlay requested but server didn't render it
            attached.append(clean_path)
            attachment_labels.append("current full clean screenshot")
        # Before/after: prepend last turn's matching frame so the model can diff
        # what this action changed (the "video" case, done as two still frames).
        prev_for_compare = prev_overlay_path if (have_overlay and visual != "clean") else prev_clean_path
        compared = compare and os.path.exists(prev_for_compare)
        if compared:
            attached = [prev_for_compare] + attached
            attachment_labels = ["previous full screenshot (before this action)"] + attachment_labels
        if not screenshot_is_blank:
            target_crops = _write_target_crops(
                clean_path, output_dir, data,
                focus_element_id=focus_element_id,
                focus_point=focus_point,
                focus_text=focus_text,
            )
            for crop_path, crop_label in target_crops:
                attached.append(crop_path)
                attachment_labels.append(crop_label)
        try:
            worker.set_visual_context(attached)
        except Exception:
            attached = []
            attachment_labels = []
            target_crops = []
            compared = False
    else:
        compared = False

    if attached:
        marks = " (with numbered [id] marks)" if any(p in (overlay_path, prev_overlay_path) for p in attached) else ""
        ordering = "; ".join(
            f"{index}) {label}" for index, label in enumerate(attachment_labels, 1))
        vision_note = (
            f"{len(attached)} image{'s are' if len(attached) != 1 else ' is'} attached to your NEXT "
            f"turn{marks}. Order: {ordering}. Use the full frame for page context and every lossless "
            "2x crop for fine text/local geometry; the crops are enlargements of the SAME current frame, "
            "not separate pages. "
            + ("Compare the previous/current full frames before acting. " if compared else "")
            + "Act only by [id] from the current INTERACTIVE ELEMENTS list."
        )
    elif not include_vision:
        vision_note = ("(screenshot NOT attached this step for speed — you are acting on the element "
                       "list alone; set include_vision=true or call browser_read to see the page.)")
    elif worker is not None and not qwen38_vision_ready:
        vision_note = (
            f"(screenshot was NOT attached: vision is restricted to {VISION_MODEL_NAME}, "
            f"but this session is using {worker_vision_model or 'an unknown model'}. "
            "Restart Aeon with the Qwen3.8 model to analyze browser screenshots.)")
    else:
        vision_note = ("(no multimodal context available to attach the screenshot; acting on the "
                       "element list. Screenshots saved to disk below.)")

    # A blank frame overrides the "look at it directly" invitation: reading anything
    # off it would be confabulation.
    if attached and screenshot_is_blank:
        vision_note = ("⚠ The attached screenshot is BLANK/UNIFORM — the page did not render into "
                       "this frame (a capture/render failure, NOT proof the page is empty). Do NOT "
                       "read, quote, or describe any on-page text from it; anything you 'see' would "
                       "be confabulated. Act only on the URL/title/element list below, or call "
                       "browser_read to re-observe. If it stays blank, report that you cannot see "
                       "the page rather than guessing its contents.")

    # Dialogs auto-handled and files downloaded since the last step are reported
    # here so the agent always knows they happened. Downloaded files are also
    # copied into the workspace ./downloads so the agent can use them directly.
    events = list(data.get("events") or [])
    events += _relocate_downloads(events)
    events_str = ("\n=== EVENTS ===\n" + "\n".join(f"- {e}" for e in events) + "\n") if events else ""

    shots = f"clean={clean_path}" + (f" | numbered={overlay_path}" if have_overlay else "")
    if target_crops:
        shots += " | targeted=" + ",".join(path for path, _ in target_crops)
    validation_str = _format_validation(data.get("validation"))
    validation_block = f"\n{validation_str}\n" if validation_str else ""
    # Ground-truth identity: the account the browser is ACTUALLY signed in as. The
    # profile is persistent+shared, so this may be a pre-existing login, NOT the
    # account you think you're working on — report/act on THIS, not your assumption.
    identity = data.get("identity")
    identity_line = (
        f"SIGNED IN AS: {identity}  (ground truth — this is the account active in the "
        f"browser right now; confirm it is the one you intend before acting on or "
        f"reporting about it)\n"
        if identity else ""
    )
    blank_block = (
        "\n⚠ BLANK SCREENSHOT — the captured frame is uniform/near-empty; the page did NOT "
        "render into it (capture/render failure, not proof the page is empty). You are flying "
        "BLIND this turn: do NOT claim to read or see any page text/content — that would be "
        "confabulation. Use only the URL/title/elements below, or browser_read to re-observe.\n"
        if screenshot_is_blank else ""
    )
    return (
        f"--- BROWSER: {action_desc} (tab '{tab_id}') ---\n"
        f"{blank_block}"
        f"{no_change_block}"
        f"{identity_line}"
        f"URL: {page_url}\n"
        f"Title: {page_title}\n"
        f"Open tabs: [{open_tabs_str}]\n"
        f"{scroll_str}\n"
        f"{events_str}"
        f"{validation_block}\n"
        f"{visible_text_block}"
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
        with _BrowserTabLock(
            payload["profile"], payload.get("session_id", "default"), tab_id
        ):
            resp = requests.post(
                f"{BROWSER_API_URL}/{endpoint}", json=payload,
                headers=browser_auth_headers(), timeout=timeout,
            )
        if resp.status_code != 200:
            # The server returns a helpful 'detail' for 4xx (e.g. expected_text mismatch).
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return f"Browser action failed ({action_desc}): HTTP {resp.status_code}: {detail}"
        return process_browser_response(resp.json(), action_desc, payload.get("session_id"),
                                        tab_id, include_vision=include_vision,
                                        visual=visual, worker=worker, compare=compare,
                                        focus_element_id=payload.get("element_id"),
                                        focus_point=((payload.get("x"), payload.get("y"))
                                                     if payload.get("x") is not None
                                                     and payload.get("y") is not None else None),
                                        focus_text=(payload.get("text") if endpoint == "find"
                                                    else payload.get("expected_text") or ""))
    except Exception as e:
        return f"Error during {action_desc}: {type(e).__name__}: {e}"


class BrowserNavigateTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_navigate", description=TOOL_DESC_BROWSER_NAVIGATE)
        self.worker = worker

    def execute(self, url: str, tab_id: str = None, include_vision: bool = True,
                visual: str = "overlay", **kwargs) -> str:
        if not url:
            return "Error: 'url' is required."
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        return _post("navigate", {"session_id": _session_id(), "tab_id": tab_id, "url": url},
                     f"Navigated to {url}", tab_id, include_vision=include_vision,
                     visual=visual, worker=self.worker)


class BrowserInteractTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_interact", description=TOOL_DESC_BROWSER_INTERACT)
        self.worker = worker

    def execute(self, action: str, element_id: int = None, text: str = None,
                expected_text: str = None, tab_id: str = None, key: str = None,
                value: str = None, file_path: str = None, amount: int = None,
                direction: str = None, to_element_id: int = None,
                clear_first: bool = True, then_enter: bool = False,
                duration: int = 2000, include_vision: bool = True,
                visual: str = "overlay", compare: bool = False,
                x: float = None, y: float = None, to_x: float = None,
                to_y: float = None, dialog_action: str = None,
                dialog_text: str = None, **kwargs) -> str:
        if not action:
            return "Error: 'action' is required."
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        action, text, duration, direction, key, value = _normalize_browser_action(
            action, text, duration, direction, key, value
        )
        if action not in _BROWSER_ACTIONS:
            allowed = ", ".join(sorted(_BROWSER_ACTIONS))
            return f"Error: unsupported browser action {action!r}. Use one of: {allowed}."

        staged_host = None
        if action == "upload_file":
            try:
                staged_host, file_path = _stage_browser_upload(file_path)
            except ValueError as exc:
                return f"Error: {exc}"

        payload = {
            "session_id": _session_id(), "tab_id": tab_id, "action": action,
            "element_id": element_id, "to_element_id": to_element_id, "text": text,
            "expected_text": expected_text, "key": key, "value": value,
            "file_path": file_path, "amount": amount, "direction": direction or "down",
            "clear_first": clear_first, "then_enter": then_enter, "duration": duration,
            "x": x, "y": y, "to_x": to_x, "to_y": to_y,
            "dialog_action": dialog_action, "dialog_text": dialog_text,
        }
        desc = f"{action}" + (f" on [{element_id}]" if element_id is not None else "")
        if action.endswith("_at") and x is not None and y is not None:
            desc += f" at ({x:g}, {y:g})"
        try:
            return _post("interact", payload, desc, tab_id, include_vision=include_vision,
                         visual=visual, worker=self.worker, compare=compare)
        finally:
            if staged_host:
                try:
                    os.unlink(staged_host)
                except OSError:
                    pass


class BrowserReadTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="browser_read",
            description=(
                "Re-observe the current page WITHOUT acting: returns a fresh indexed element list and "
                "scroll state, and attaches the current page screenshot to your next turn so you SEE the "
                "page again. Use it after a page updates on its own, when element ids feel stale, or to "
                "re-orient (take a fresh look) before choosing the next action.\n"
                "Schema:\n  tab_id (str, optional): the tab to read; OMIT to read the tab you last "
                "navigated/acted in (the usual case). Only set it to read a DIFFERENT open tab.\n"
                "  include_vision (bool, optional, default true): set false to skip attaching the "
                "screenshot for a faster, element-list-only read when you don't need to see the page.\n"
                "  visual (str, optional, default 'overlay'): 'overlay' = render + numbered [id] marks; "
                "'clean' = pure render, no marks; 'both'.\n"
                "Example: {\"tool_name\": \"browser_read\", \"parameters\": {\"tab_id\": \"gmail\"}}"
            ),
        )
        self.worker = worker

    def execute(self, tab_id: str = None, include_vision: bool = True,
                visual: str = "overlay", **kwargs) -> str:
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        return _post("observe", {"session_id": _session_id(), "tab_id": tab_id}, "Read page", tab_id,
                     include_vision=include_vision, visual=visual, worker=self.worker)


class BrowserFindTool(BaseTool):
    """Search the rendered page semantically and return actionable matches."""

    def __init__(self, worker=None):
        super().__init__(
            name="browser_find",
            description=(
                "Find text or controls anywhere in the current rendered page, including off-screen "
                "content, iframes, and open shadow roots. Returns a filtered indexed element list "
                "and a numbered screenshot; matches can be used immediately with browser_interact. "
                "Schema: text (required substring), role (optional accessibility role), tab_id "
                "(optional), include_vision (default true), visual ('overlay'|'clean'|'both')."
            ),
        )
        self.worker = worker

    def execute(self, text: str, role: str = None, tab_id: str = None,
                include_vision: bool = True, visual: str = "overlay", **kwargs) -> str:
        if not text or not str(text).strip():
            return "Error: browser_find requires non-empty 'text'."
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        payload = {
            "session_id": _session_id(), "tab_id": tab_id,
            "text": str(text).strip(), "role": str(role or "").strip() or None,
        }
        desc = f"Find {text!r}" + (f" with role {role!r}" if role else "")
        return _post("find", payload, desc, tab_id, include_vision=include_vision,
                     visual=visual, worker=self.worker)


class BrowserExtractTool(BaseTool):
    """Extract structured page content without forcing the model to OCR it."""

    def __init__(self, worker=None):
        super().__init__(
            name="browser_extract",
            description=(
                "Extract structured content from the current rendered page and its iframes. "
                "Use mode='forms' to inspect controls/labels/options, mode='tables' for headers "
                "and rows, mode='links' for link text and resolved URLs, or mode='text' for main "
                "page text. Schema: mode (required), tab_id (optional), max_items (default 200, "
                "maximum 500). This is read-only and returns grounded text/JSON without a screenshot."
            ),
        )
        self.worker = worker

    def execute(self, mode: str, tab_id: str = None, max_items: int = 200,
                **kwargs) -> str:
        mode = str(mode or "").strip().lower()
        if mode not in {"text", "links", "forms", "tables"}:
            return "Error: browser_extract mode must be text, links, forms, or tables."
        try:
            max_items = max(1, min(int(max_items), 500))
        except (TypeError, ValueError):
            return "Error: browser_extract max_items must be an integer."
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        payload = {
            "session_id": _session_id(), "tab_id": tab_id,
            "mode": mode, "max_items": max_items,
        }
        return _post(
            "extract", payload, f"Extract {mode}", tab_id,
            include_vision=False, visual="clean", worker=self.worker,
        )


class BrowserCloseTabTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_close_tab", description=TOOL_DESC_BROWSER_CLOSE_TAB)
        self.worker = worker

    def execute(self, tab_id: str) -> str:
        try:
            ensure_browser_running()
            profile = _profile_for(self.worker)
            with _BrowserTabLock(profile, _session_id(), tab_id):
                resp = requests.post(f"{BROWSER_API_URL}/close_tab",
                                     json={"session_id": _session_id(), "tab_id": tab_id,
                                           "profile": profile},
                                     headers=browser_auth_headers(), timeout=15)
            if resp.status_code != 200:
                return f"HTTP Error {resp.status_code} from browser API: {resp.text}"
            data = resp.json()
            # If we just closed the remembered "last used" tab, forget it —
            # otherwise the next omitted tab_id resolves to the closed tab and
            # 404s (the exact footgun _resolve_tab exists to prevent).
            if self.worker is not None and getattr(self.worker, "_last_browser_tab", None) == tab_id:
                self.worker._last_browser_tab = None
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
        # A switch names an explicit target; remember it so later omitted-tab
        # interacts follow the tab the agent switched to.
        _remember_tab(self.worker, tab_id)
        return _post("switch_tab", {"session_id": _session_id(), "tab_id": tab_id},
                     f"Switched to tab '{tab_id}'", tab_id,
                     include_vision=include_vision, visual=visual, worker=self.worker)


class BrowserCaptureMediaTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(name="browser_capture_media", description=TOOL_DESC_BROWSER_CAPTURE_MEDIA)
        self.worker = worker

    def execute(self, output_dir: str = None, media_id: int = None,
                tab_id: str = None, duration_s: int = 8, **kwargs) -> str:
        tab_id = _resolve_tab(self.worker, tab_id)
        _remember_tab(self.worker, tab_id)
        try:
            ensure_browser_running()
            payload = {"session_id": _session_id(), "tab_id": tab_id,
                       "profile": _profile_for(self.worker), "media_id": media_id,
                       "duration_s": duration_s}
            # Generous timeout: a video screen-recording fallback runs for duration_s.
            with _BrowserTabLock(payload["profile"], payload["session_id"], tab_id):
                resp = requests.post(
                    f"{BROWSER_API_URL}/capture_media", json=payload,
                    headers=browser_auth_headers(), timeout=300,
                )
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                return f"Browser capture failed: HTTP {resp.status_code}: {detail}"
            data = resp.json()
        except Exception as e:
            return f"Error capturing media: {type(e).__name__}: {e}"

        # No id chosen -> hand back the catalog so the model can pick one.
        if data.get("mode") == "list":
            return _format_media_list(data)

        # Capture mode: the service saved the file into the shared download dir;
        # move it into the caller's requested output_dir.
        if not output_dir or not str(output_dir).strip():
            return ("Error: 'output_dir' is required to save the media — the directory to write the "
                    "file into (e.g. '.' for the current workspace).")
        filename = data.get("filename")
        src = os.path.join(_host_download_dir(), filename)
        for _ in range(15):  # bind-mount flush can lag the service's write
            if os.path.exists(src):
                break
            time.sleep(0.2)
        if not os.path.exists(src):
            return (f"Capture reported success ({data.get('method')}) but the file did not appear "
                    f"at {src}. The browser service may be running without the shared profile mount.")
        dest = str(resolve_output_dir(output_dir, filename))
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            return f"Captured to {src} but could not copy it into {output_dir}: {e}"
        # The service writes as root (docker), so we may not be able to delete the
        # source — best-effort cleanup; the copy above is what matters.
        try:
            os.remove(src)
        except Exception:
            pass
        size = os.path.getsize(dest)
        return (f"Captured {data.get('tag')} (media_id {data.get('media_id')}) via "
                f"{data.get('method')}. Saved to: {dest} ({size:,} bytes).")
