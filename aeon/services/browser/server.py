"""
Aeon browser service — a human-grade web agent backend.

Design (rewritten):
  * REAL Google Chrome (not Chromium) driven by Patchright (a patched, API-compatible
    Playwright that removes the CDP automation tells), running HEADED under the
    container's Xvfb display as a PERSISTENT context (user-data-dir) so logins/cookies
    survive. No spoofed UA/viewport and no detectable evasion shims — combined with
    human-like mouse/keyboard input and the host's residential IP, it aims to be
    indistinguishable from a person at a normal browser.
  * Every observation builds a STABLE, semantic element index: each visible
    interactable/meaningful node is stamped with a `data-aeon-id` attribute (so an
    action targets the EXACT node, never a guessed CSS selector) and described by
    its accessibility role, accessible name, value, and state (expanded/selected/
    checked/disabled/editable). This is the "screen-reader" view the agent reads.
  * A REAL Set-of-Mark overlay is drawn (numbered boxes over in-viewport indexed
    elements) and screenshotted, so the vision model's "box N" is grounded in the
    same index the agent acts on.
  * A full human action vocabulary, all addressed by element index:
    click/double_click/right_click, hover, type (real keystrokes), press_key,
    scroll (page or within an element's scroll container, by amount/to-element/
    to-end), drag, press_and_hold, select_option, check/uncheck, clear,
    upload_file, go_back/go_forward/reload, wait_for, get_text.
  * Multi-tab + popup capture (OAuth windows), verified clicks (expected_text),
    and fast, non-hanging waits (domcontentloaded + bounded settle).

The agent never relies on a single lossy channel: each response returns the
structured element snapshot AND the annotated screenshot.
"""
import asyncio
import base64
import os
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Patchright is a drop-in, API-compatible Playwright fork that PATCHES the
# automation tells real bot-detection probes for: it avoids the CDP
# `Runtime.enable` leak, removes the console/runtime fingerprints, and does not
# inject detectable evasion shims. Combined with real headed Chrome + a
# persistent profile + human-like input, this is about as close to a real human
# at a real browser as we can get. (Import name mirrors playwright exactly.)
from patchright.async_api import async_playwright, Page, BrowserContext, Error as PWError

app = FastAPI()

# --- Configuration ----------------------------------------------------------
PROFILE_DIR = os.environ.get("AEON_BROWSER_PROFILE", "/profiles/default")
TIMEZONE = os.environ.get("AEON_BROWSER_TZ", "America/Los_Angeles")
LOCALE = os.environ.get("AEON_BROWSER_LOCALE", "en-US")
MAX_INDEXED_ELEMENTS = 250
MAX_FRAMES = 30               # cap frames scanned per observation
ACTION_SETTLE_MS = 500        # let the DOM react after an action before re-reading
DOWNLOAD_DIR = os.environ.get("AEON_BROWSER_DOWNLOADS", "/profiles/downloads")

# --- Global state -----------------------------------------------------------
_playwright = None
context: Optional[BrowserContext] = None
# page_key -> Page  (page_key = "<session_id>::<tab_id>")
pages: Dict[str, Page] = {}
# page_key -> {aeon_id: element_info}  (last observed snapshot, for verification)
last_elements: Dict[str, Dict[int, Dict[str, Any]]] = {}
# page_key -> {aeon_id: Frame}  (which frame owns each id, for frame-aware actions)
frame_maps: Dict[str, Dict[int, Any]] = {}
# id(page) -> list of recent dialog/download notes to surface in the next obs
page_events: Dict[int, List[str]] = {}
_popup_counter = 0


def _key(session_id: str, tab_id: str) -> str:
    return f"{session_id}::{tab_id}"


def _tabs_for(session_id: str) -> List[str]:
    prefix = f"{session_id}::"
    return [k[len(prefix):] for k in pages if k.startswith(prefix)]


# ============================================================================
# Injected JavaScript: element extraction + Set-of-Mark overlay
# ============================================================================

# Builds the stable element index. Stamps data-aeon-id on each visible
# interactable/meaningful node and returns role/name/value/state/geometry.
_EXTRACT_JS = r"""
({maxEls, startId}) => {
  // Clear any previous marks so ids are reassigned deterministically.
  document.querySelectorAll('[data-aeon-id]').forEach(e => e.removeAttribute('data-aeon-id'));

  const INTERACTIVE = [
    'a[href]','button','input','select','textarea','summary','details','label',
    '[role=button]','[role=link]','[role=checkbox]','[role=radio]','[role=tab]',
    '[role=menuitem]','[role=option]','[role=switch]','[role=textbox]','[role=combobox]',
    '[role=searchbox]','[role=slider]','[role=spinbutton]','[role=treeitem]',
    '[onclick]','[contenteditable=""]','[contenteditable=true]','[tabindex]'
  ].join(',');
  // Semantic, non-interactive containers worth surfacing (e.g. inbox rows).
  const SEMANTIC = '[role=row],[role=listitem],[role=article],[role=heading],[role=gridcell],[role=cell]';
  const MATCH = INTERACTIVE + ',' + SEMANTIC;

  function implicitRole(el) {
    const tag = el.tagName.toLowerCase();
    if (tag === 'a' && el.hasAttribute('href')) return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'summary') return 'disclosure';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      if (t === 'submit' || t === 'button') return 'button';
      if (t === 'search') return 'searchbox';
      return 'textbox';
    }
    return tag;
  }
  function accName(el) {
    let n = el.getAttribute('aria-label');
    if (n && n.trim()) return n.trim();
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(id => { const e = document.getElementById(id); return e ? e.innerText : ''; }).join(' ').trim();
      if (t) return t;
    }
    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map(l => l.innerText).join(' ').trim();
      if (t) return t;
    }
    let t = (el.innerText || '').trim();
    if (t) return t;
    const ph = el.getAttribute('placeholder'); if (ph) return ph.trim();
    if (el.value) return String(el.value).trim();
    const alt = el.getAttribute('alt'); if (alt) return alt.trim();
    const title = el.getAttribute('title'); if (title) return title.trim();
    return '';
  }
  function states(el) {
    const out = [];
    const exp = el.getAttribute('aria-expanded');
    if (exp === 'true') out.push('expanded'); else if (exp === 'false') out.push('collapsed');
    const sel = el.getAttribute('aria-selected');
    if (sel === 'true') out.push('selected');
    const chk = el.getAttribute('aria-checked');
    if (chk === 'true' || el.checked === true) out.push('checked');
    if (el.getAttribute('aria-current')) out.push('current');
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') out.push('disabled');
    if (el.isContentEditable || ['input','textarea','select'].includes(el.tagName.toLowerCase())) out.push('editable');
    if (el.open === true) out.push('open');
    return out;
  }
  function scrollableAncestor(el) {
    let p = el.parentElement;
    while (p && p !== document.body) {
      const s = window.getComputedStyle(p);
      const oy = s.overflowY;
      if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight + 4) return p;
      p = p.parentElement;
    }
    return null;
  }

  // Recursive walk that PIERCES open shadow roots (querySelectorAll does not),
  // reading each candidate's style+rect exactly once (no reflow storm). Closed
  // shadow roots are inaccessible (rare). Cross-document iframes are handled in
  // Python by running this same script in every frame.
  const seen = new Set();
  const cands = [];
  function consider(el) {
    if (seen.has(el)) return;
    seen.add(el);
    let ok = false;
    try { ok = el.matches && el.matches(MATCH); } catch (e) { ok = false; }
    if (ok) {
      const s = window.getComputedStyle(el);
      if (!(s.visibility === 'hidden' || s.display === 'none' || parseFloat(s.opacity || '1') === 0)) {
        const r = el.getBoundingClientRect();
        if (r.width >= 2 && r.height >= 2) cands.push({ el: el, r: r });
      }
    }
  }
  function walk(root) {
    let node = root.firstElementChild;
    while (node) {
      consider(node);
      if (node.shadowRoot) walk(node.shadowRoot);
      walk(node);
      node = node.nextElementSibling;
    }
  }
  walk(document);

  // Sort top-to-bottom, left-to-right on the precomputed rects (no reflow).
  cands.sort((a, b) => {
    const dy = a.r.top - b.r.top;
    if (Math.abs(dy) > 8) return dy;
    return a.r.left - b.r.left;
  });

  const results = [];
  let n = 0;
  let scId = 0;
  const scMap = new Map();
  for (const cand of cands) {
    if (n >= maxEls) break;
    const el = cand.el, r = cand.r;
    const id = startId + n;
    n += 1;
    el.setAttribute('data-aeon-id', String(id));
    const sc = scrollableAncestor(el);
    let scKey = null;
    if (sc) {
      if (!scMap.has(sc)) { scId += 1; scMap.set(sc, scId); sc.setAttribute('data-aeon-sc', String(scId)); }
      scKey = scMap.get(sc);
    }
    let name = accName(el);
    if (name.length > 160) name = name.slice(0, 160) + '…';
    results.push({
      id: id,
      role: el.getAttribute('role') || implicitRole(el),
      name: name,
      value: (el.value !== undefined && el.value !== null) ? String(el.value).slice(0, 120) : '',
      states: states(el),
      rect: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
      scrollContainer: scKey
    });
  }

  const de = document.scrollingElement || document.documentElement;
  return {
    elements: results,
    nextId: startId + n,
    page: {
      scrollY: Math.round(window.scrollY),
      scrollHeight: Math.round(de.scrollHeight),
      clientHeight: Math.round(de.clientHeight || window.innerHeight),
      innerWidth: window.innerWidth,
      innerHeight: window.innerHeight
    }
  };
}
"""

# Draws numbered Set-of-Mark boxes from an ABSOLUTE-coordinate list (computed in
# Python with per-frame offsets), so elements inside iframes/shadow DOM get boxes
# too — not just main-document elements.
_OVERLAY_JS = r"""
(items) => {
  const old = document.getElementById('__aeon_som__');
  if (old) old.remove();
  const layer = document.createElement('div');
  layer.id = '__aeon_som__';
  layer.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;z-index:2147483647;pointer-events:none;';
  const palette = ['#e6194B','#3cb44b','#4363d8','#f58231','#911eb4','#008080','#9A6324','#800000'];
  let drawn = 0;
  for (const it of items) {
    const c = palette[it.id % palette.length];
    const box = document.createElement('div');
    box.style.cssText = 'position:fixed;border:2px solid ' + c + ';left:' + it.x + 'px;top:' + it.y +
      'px;width:' + it.w + 'px;height:' + it.h + 'px;box-sizing:border-box;pointer-events:none;';
    const tag = document.createElement('div');
    tag.textContent = it.id;
    tag.style.cssText = 'position:fixed;left:' + it.x + 'px;top:' + Math.max(0, it.y - 14) +
      'px;background:' + c + ';color:#fff;font:bold 11px/1.1 monospace;padding:0 3px;pointer-events:none;';
    layer.appendChild(box); layer.appendChild(tag); drawn++;
  }
  document.body.appendChild(layer);
  return drawn;
}
"""

_OVERLAY_CLEAR_JS = "() => { const o = document.getElementById('__aeon_som__'); if (o) o.remove(); }"

# Readability: pick the main-content root and return title + clean text. Used by
# the read_text action for reading/extraction tasks (the structured snapshot only
# surfaces interactive/semantic elements, not full article prose).
_READABILITY_JS = r"""
() => {
  function tlen(el){ return ((el.innerText||'').replace(/\s+/g,' ').trim()).length; }
  let root = document.querySelector('article, main, [role=main]');
  if (!root) {
    let best = document.body, bestScore = 0;
    document.querySelectorAll('div, section, article, main, td').forEach(el => {
      let score = 0;
      el.querySelectorAll('p, li').forEach(p => score += tlen(p));
      if (score > bestScore) { bestScore = score; best = el; }
    });
    root = best || document.body;
  }
  const title = (document.title || '').trim();
  let text = ((root && root.innerText) || document.body.innerText || '').replace(/\n{3,}/g, '\n\n').trim();
  return (title ? ('# ' + title + '\n\n') : '') + text;
}
"""


# ============================================================================
# Browser lifecycle
# ============================================================================

async def _ensure_browser():
    """Launch the most human-indistinguishable browser we can.

    Maximum realism, in order of importance:
      1. REAL Google Chrome (channel='chrome') — correct branding, build flags,
         fonts, codecs and TLS/JA3 fingerprint, not open-source Chromium (which
         leaks 'HeadlessChrome'/Chromium build signals).
      2. HEADED under Xvfb — headless has dozens of detectable differences.
      3. Patchright's patched driver — no CDP `Runtime.enable` leak, the single
         biggest automation tell; it also injects NO detectable evasion shim, so
         we deliberately do NOT add manual navigator.webdriver hacks (those are
         themselves fingerprinted).
      4. NO user_agent / viewport override — the real browser reports a perfectly
         self-consistent UA + userAgentData + window size; spoofing either
         creates a mismatch detectors look for. We use the real window (no_viewport).
      5. Persistent profile — real cookies/history/login state, like a person's
         everyday browser.
    Behavioral realism (eased mouse paths, real per-key typing delays) is in the
    action helpers. The container uses the host's residential IP.
    """
    global _playwright, context
    if context is not None:
        return context
    os.makedirs(PROFILE_DIR, exist_ok=True)
    _playwright = await async_playwright().start()
    launch_kwargs = dict(
        user_data_dir=PROFILE_DIR,
        headless=False,
        no_viewport=True,           # use the real OS window size, no viewport tell
        accept_downloads=True,      # so the download handler can save files
        locale=LOCALE,
        timezone_id=TIMEZONE,
        # Keep args minimal: patchright strips the automation flags. --no-sandbox
        # is required to run Chrome as root in a container; the rest aid Xvfb.
        args=["--no-sandbox", "--disable-dev-shm-usage", "--start-maximized"],
    )
    try:
        context = await _playwright.chromium.launch_persistent_context(channel="chrome", **launch_kwargs)
    except Exception as e:
        # Fall back to the patched Chromium if real Chrome isn't present.
        print(f"[browser] real Chrome channel unavailable ({e}); using patched Chromium.")
        context = await _playwright.chromium.launch_persistent_context(**launch_kwargs)
    # Capture popups/new tabs (OAuth windows, target=_blank) automatically.
    context.on("page", _register_popup)
    # Crash recovery: if the browser/context dies, drop our state so the next
    # request transparently relaunches instead of wedging on a dead context.
    context.on("close", _on_context_close)
    return context


def _on_context_close():
    global context
    context = None
    pages.clear()
    frame_maps.clear()
    last_elements.clear()
    page_events.clear()


def _note_event(page: Page, msg: str):
    page_events.setdefault(id(page), []).append(msg)


def _setup_page(page: Page):
    """Attach dialog / download / crash handlers to a page we manage."""
    page.on("dialog", lambda d: asyncio.create_task(_handle_dialog(page, d)))
    page.on("download", lambda d: asyncio.create_task(_handle_download(page, d)))
    page.on("crash", lambda: _note_event(page, "[page crashed — reload or re-navigate]"))


async def _handle_dialog(page: Page, dialog):
    # Native alert/confirm/prompt/beforeunload would otherwise block the next
    # action. Accept by default (so navigation proceeds) and report it so the
    # agent knows it happened.
    _note_event(page, f"[dialog:{dialog.type}] {dialog.message[:200]} (auto-accepted)")
    try:
        await dialog.accept()
    except Exception:
        try:
            await dialog.dismiss()
        except Exception:
            pass


async def _handle_download(page: Page, download):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    name = (download.suggested_filename or "download").replace("/", "_")
    dest = os.path.join(DOWNLOAD_DIR, name)
    try:
        await download.save_as(dest)
        host = dest.replace("/profiles", "~/.aeon/browser_profiles", 1)
        _note_event(page, f"[download] saved '{name}' -> {host}")
    except Exception as e:
        _note_event(page, f"[download] failed for '{name}': {e}")


def _register_popup(page: Page):
    # The "page" event also fires for pages WE create via new_page(); defer to an
    # async task that checks opener() so we only capture genuine popups (OAuth /
    # target=_blank windows), never our own tabs, and avoid a registration race.
    asyncio.create_task(_maybe_register_popup(page))


async def _maybe_register_popup(page: Page):
    global _popup_counter
    try:
        opener = await page.opener()
    except Exception:
        opener = None
    if opener is None:
        return  # one of our own new_page() tabs, not a popup
    if page in pages.values():
        return
    _setup_page(page)
    _popup_counter += 1
    pages[_key("_popups", f"popup_{_popup_counter}")] = page


async def _get_page(session_id: str, tab_id: str) -> Page:
    k = _key(session_id, tab_id)
    if k not in pages:
        raise HTTPException(status_code=404, detail=f"Tab '{tab_id}' not found. Navigate to open it.")
    return pages[k]


# ============================================================================
# Human-like input helpers
# ============================================================================

async def _human_pause(lo=0.05, hi=0.18):
    await asyncio.sleep(random.uniform(lo, hi))


async def _human_move_to(page: Page, x: float, y: float):
    """Move the cursor to (x, y) in several eased steps with slight jitter."""
    await page.mouse.move(x + random.uniform(-1, 1), y + random.uniform(-1, 1),
                          steps=random.randint(8, 18))


def _frame_for(session_id: str, tab_id: str, aeon_id: int, page: Page):
    """The frame that owns this id (so iframe/shadow elements resolve correctly);
    falls back to the main frame."""
    return frame_maps.get(_key(session_id, tab_id), {}).get(aeon_id) or page.main_frame


async def _locator(page: Page, aeon_id: int, session_id: str = None, tab_id: str = None):
    owner = _frame_for(session_id, tab_id, aeon_id, page) if session_id is not None else page
    return owner.locator(f'[data-aeon-id="{aeon_id}"]').first


async def _center(page: Page, locator) -> Optional[Dict[str, float]]:
    try:
        box = await locator.bounding_box()
    except PWError:
        box = None
    if not box:
        return None
    return {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}


async def _human_click(page: Page, locator, button: str = "left", clicks: int = 1):
    await locator.scroll_into_view_if_needed(timeout=5000)
    c = await _center(page, locator)
    if c:
        await _human_move_to(page, c["x"], c["y"])
        await _human_pause()
        await page.mouse.click(c["x"], c["y"], button=button, click_count=clicks,
                               delay=random.randint(40, 110))
    else:
        # Fallback to element click if geometry is unavailable.
        await locator.click(button=button, click_count=clicks, timeout=5000)


async def _human_type(page: Page, locator, text: str, clear_first: bool, then_enter: bool):
    await _human_click(page, locator)
    await _human_pause()
    if clear_first:
        try:
            await locator.fill("")
        except PWError:
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Delete")
    for ch in text:
        await page.keyboard.type(ch, delay=random.randint(20, 90))
    if then_enter:
        await _human_pause()
        await page.keyboard.press("Enter")


async def _press_and_hold(page: Page, locator, duration_ms: int):
    await locator.scroll_into_view_if_needed(timeout=5000)
    c = await _center(page, locator)
    if not c:
        raise HTTPException(status_code=500, detail="Could not locate element to press-and-hold.")
    await _human_move_to(page, c["x"], c["y"])
    await page.mouse.down()
    start = time.time()
    while (time.time() - start) * 1000 < duration_ms:
        await page.mouse.move(c["x"] + random.uniform(-2, 2), c["y"] + random.uniform(-2, 2))
        await asyncio.sleep(0.05)
    await page.mouse.up()


# ============================================================================
# Observation (screenshots + structured snapshot)
# ============================================================================

async def _settle_nav(page: Page):
    """Wait for a real page load WITHOUT hanging. domcontentloaded is the true
    signal; networkidle is only a short, bounded bonus because SPAs that keep a
    socket open (Gmail, chat apps) never reach networkidle and would otherwise
    burn the full timeout every navigation."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=8000)
    except PWError:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=1500)
    except PWError:
        pass
    await asyncio.sleep(0.2)


async def _settle_action(page: Page):
    """Light settle after an in-page interaction: if the action navigated,
    domcontentloaded fires; if it only mutated the DOM, this returns ~instantly.
    Deliberately NO networkidle wait — that is the main per-action latency sink."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=4000)
    except PWError:
        pass
    await asyncio.sleep(ACTION_SETTLE_MS / 1000)


async def _frame_offset(page: Page, frame) -> Optional[Dict[str, float]]:
    """Top-left of a frame in the MAIN page's viewport coordinates (0,0 for the
    main frame). Returns None if the frame is detached/unmeasurable so we skip it."""
    if frame == page.main_frame:
        return {"x": 0.0, "y": 0.0, "iw": 0, "ih": 0}
    try:
        fe = await frame.frame_element()
        box = await fe.bounding_box()
    except Exception:
        return None
    if not box:
        return None
    return {"x": box["x"], "y": box["y"], "iw": 0, "ih": 0}


async def _extract(page: Page, session_id: str, tab_id: str) -> Dict[str, Any]:
    """Run the extractor in EVERY frame (main + same/cross-origin iframes), each
    piercing its own shadow DOM, and merge into one globally-numbered element list
    with absolute (main-viewport) coordinates. Tracks which frame owns each id so
    actions can target the right frame."""
    key = _key(session_id, tab_id)
    elements: List[Dict[str, Any]] = []
    fmap: Dict[int, Any] = {}
    main_state = {"scrollY": 0, "scrollHeight": 0, "clientHeight": 0, "innerWidth": 1280, "innerHeight": 1024}

    start = 1
    for frame in page.frames[:MAX_FRAMES]:
        if start - 1 >= MAX_INDEXED_ELEMENTS:
            break
        off = await _frame_offset(page, frame)
        if off is None:
            continue
        try:
            data = await frame.evaluate(_EXTRACT_JS, {"maxEls": MAX_INDEXED_ELEMENTS - (start - 1),
                                                      "startId": start})
        except Exception:
            continue
        if frame == page.main_frame:
            main_state = data.get("page", main_state)
        ox, oy = off["x"], off["y"]
        iw, ih = main_state["innerWidth"], main_state["innerHeight"]
        for e in data["elements"]:
            r = e["rect"]
            ax, ay = r["x"] + ox, r["y"] + oy
            e["rect"] = {"x": round(ax), "y": round(ay), "w": r["w"], "h": r["h"]}
            e["inViewport"] = (ay + r["h"] > 0 and ax + r["w"] > 0 and ay < ih and ax < iw)
            if frame != page.main_frame:
                e["inFrame"] = True
            fmap[e["id"]] = frame
            elements.append(e)
        start = data.get("nextId", start)

    last_elements[key] = {e["id"]: e for e in elements}
    frame_maps[key] = fmap
    return {"elements": elements, "page": main_state}


async def _build_response(page: Page, session_id: str, tab_id: str) -> Dict[str, Any]:
    extracted = await _extract(page, session_id, tab_id)

    # Clean viewport screenshot (readable, real resolution) + Set-of-Mark overlay
    # drawn from the merged absolute coordinates (covers iframe/shadow elements).
    clean_bytes = await page.screenshot(type="jpeg", quality=80)
    overlay_items = [
        {"id": e["id"], "x": e["rect"]["x"], "y": e["rect"]["y"], "w": e["rect"]["w"], "h": e["rect"]["h"]}
        for e in extracted["elements"] if e.get("inViewport")
    ]
    overlay_bytes = clean_bytes
    try:
        await page.evaluate(_OVERLAY_JS, overlay_items)
        overlay_bytes = await page.screenshot(type="jpeg", quality=80)
    except PWError:
        overlay_bytes = clean_bytes
    finally:
        try:
            await page.evaluate(_OVERLAY_CLEAR_JS)
        except PWError:
            pass

    title = await page.title()
    # Drain any dialog/download notes captured since the last observation.
    events = page_events.pop(id(page), [])
    return {
        "status": "success",
        "session_id": session_id,
        "tab_id": tab_id,
        "url": page.url,
        "title": title,
        "clean_b64": base64.b64encode(clean_bytes).decode(),
        "overlay_b64": base64.b64encode(overlay_bytes).decode(),
        "elements": extracted["elements"],
        "page_state": extracted["page"],
        "events": events,
        "open_tabs": _tabs_for(session_id) + [t for t in _tabs_for("_popups")],
    }


# ============================================================================
# Request models
# ============================================================================

class NavigateRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    url: str


class TabRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"


class InteractRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    action: str
    element_id: Optional[int] = None
    to_element_id: Optional[int] = None     # drag target
    text: Optional[str] = None
    expected_text: Optional[str] = None
    key: Optional[str] = None               # for press_key
    value: Optional[str] = None             # for select_option
    file_path: Optional[str] = None         # for upload_file
    amount: Optional[int] = None            # scroll pixels
    direction: Optional[str] = "down"       # scroll direction
    clear_first: Optional[bool] = True      # for type
    then_enter: Optional[bool] = False      # for type
    duration: Optional[int] = 2000          # press_and_hold ms


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ping")
async def ping():
    return {"status": "pong", "version": "human_v2"}


@app.post("/navigate")
async def navigate(req: NavigateRequest):
    ctx = await _ensure_browser()
    k = _key(req.session_id, req.tab_id)
    if k not in pages:
        page = await ctx.new_page()
        _setup_page(page)
        pages[k] = page
    page = pages[k]
    url = req.url if "://" in req.url else "https://" + req.url
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except PWError as e:
        return {"status": "error", "msg": f"Navigation to {url} failed: {e}"}
    await _settle_nav(page)
    return await _build_response(page, req.session_id, req.tab_id)


def _verify_expected(session_id: str, tab_id: str, element_id: int, expected: str) -> Optional[str]:
    """Return an error string if expected_text doesn't match the indexed element."""
    if not expected:
        return None
    snap = last_elements.get(_key(session_id, tab_id), {})
    info = snap.get(element_id)
    if not info:
        return None  # no snapshot to check against; allow
    hay = f"{info.get('name', '')} {info.get('value', '')}".lower()
    if expected.lower() not in hay:
        return (f"expected_text '{expected}' not found in element [{element_id}] "
                f"(role={info.get('role')}, name='{info.get('name')}'). "
                f"Re-read the page and target the correct element.")
    return None


@app.post("/interact")
async def interact(req: InteractRequest):
    ctx = await _ensure_browser()
    page = await _get_page(req.session_id, req.tab_id)
    a = (req.action or "").strip().lower()

    # ---- page-level actions that need no element ----
    try:
        if a in ("go_back", "back"):
            await page.go_back(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, req.session_id, req.tab_id)
        if a in ("go_forward", "forward"):
            await page.go_forward(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, req.session_id, req.tab_id)
        if a in ("reload", "refresh"):
            await page.reload(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, req.session_id, req.tab_id)
        if a == "press_key":
            if not req.key:
                raise HTTPException(status_code=400, detail="press_key requires 'key'.")
            if req.element_id is not None:
                await (await _locator(page, req.element_id, req.session_id, req.tab_id)).press(req.key)
            else:
                await page.keyboard.press(req.key)
            await asyncio.sleep(ACTION_SETTLE_MS / 1000)
            return await _build_response(page, req.session_id, req.tab_id)
        if a == "scroll":
            await _do_scroll(page, req)
            await asyncio.sleep(ACTION_SETTLE_MS / 1000)
            return await _build_response(page, req.session_id, req.tab_id)
        if a == "wait_for":
            await _do_wait(page, req)
            return await _build_response(page, req.session_id, req.tab_id)
        if a == "read_text":
            # Clean main-content text (readability), for reading/extraction tasks.
            try:
                txt = await page.evaluate(_READABILITY_JS)
            except PWError:
                txt = await page.inner_text("body")
            return {"text": (txt or "")[:14000]}
        if a in ("get_page_text", "get_text") and req.element_id is None:
            return {"text": (await page.inner_text("body"))[:14000]}

        # ---- element-targeted actions ----
        if req.element_id is None:
            raise HTTPException(status_code=400, detail=f"action '{a}' requires element_id.")

        verr = _verify_expected(req.session_id, req.tab_id, req.element_id, req.expected_text or "")
        if verr and a in ("click", "double_click", "right_click"):
            raise HTTPException(status_code=409, detail=verr)

        loc = await _locator(page, req.element_id, req.session_id, req.tab_id)
        try:
            await loc.wait_for(state="attached", timeout=4000)
        except PWError:
            raise HTTPException(status_code=404,
                                detail=f"Element [{req.element_id}] is no longer on the page. Re-read it.")

        if a == "click":
            await _human_click(page, loc)
        elif a == "double_click":
            await _human_click(page, loc, clicks=2)
        elif a == "right_click":
            await _human_click(page, loc, button="right")
        elif a == "hover":
            await loc.scroll_into_view_if_needed(timeout=5000)
            c = await _center(page, loc)
            if c:
                await _human_move_to(page, c["x"], c["y"])
            else:
                await loc.hover()
        elif a == "type":
            await _human_type(page, loc, req.text or "", bool(req.clear_first), bool(req.then_enter))
        elif a == "press_and_hold":
            await _press_and_hold(page, loc, req.duration or 2000)
        elif a == "select_option":
            await loc.select_option(req.value)
        elif a == "check":
            await loc.check()
        elif a == "uncheck":
            await loc.uncheck()
        elif a == "clear":
            await loc.fill("")
        elif a == "upload_file":
            if not req.file_path:
                raise HTTPException(status_code=400, detail="upload_file requires 'file_path'.")
            await _do_upload(page, loc, req.file_path)
        elif a == "drag":
            if req.to_element_id is None:
                raise HTTPException(status_code=400, detail="drag requires 'to_element_id'.")
            await _do_drag(page, req.element_id, req.to_element_id, req.session_id, req.tab_id)
        elif a == "get_text":
            return {"text": await loc.inner_text()}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {a}")

        await _settle_action(page)
        return await _build_response(page, req.session_id, req.tab_id)

    except HTTPException:
        raise
    except PWError as e:
        raise HTTPException(status_code=500, detail=f"Browser action '{a}' failed: {e}")


async def _do_scroll(page: Page, req: InteractRequest):
    amount = req.amount if req.amount is not None else 600
    direction = (req.direction or "down").lower()
    if req.element_id is not None and direction in ("to_element", "into_view"):
        await (await _locator(page, req.element_id, req.session_id, req.tab_id)).scroll_into_view_if_needed(timeout=5000)
        return
    if req.element_id is not None:
        # Scroll the scroll-container holding the element — IN ITS OWN FRAME, so
        # an inbox/list/chat pane inside an iframe scrolls correctly.
        owner = _frame_for(req.session_id, req.tab_id, req.element_id, page)
        sign = 1 if direction in ("down", "bottom") else -1
        await owner.evaluate(
            """({id, delta, toEnd}) => {
                const el = document.querySelector('[data-aeon-id=\"' + id + '\"]');
                if (!el) return;
                let p = el.parentElement;
                while (p && p !== document.body) {
                    const s = getComputedStyle(p);
                    if ((s.overflowY === 'auto' || s.overflowY === 'scroll') && p.scrollHeight > p.clientHeight + 4) {
                        if (toEnd) { p.scrollTop = delta > 0 ? p.scrollHeight : 0; }
                        else { p.scrollTop += delta; }
                        return;
                    }
                    p = p.parentElement;
                }
                window.scrollBy(0, delta);
            }""",
            {"id": req.element_id, "delta": sign * abs(amount),
             "toEnd": direction in ("top", "bottom")},
        )
        return
    # Page-level scroll.
    if direction == "bottom":
        await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    elif direction == "top":
        await page.evaluate("() => window.scrollTo(0, 0)")
    else:
        dy = abs(amount) * (1 if direction == "down" else -1)
        await page.mouse.wheel(0, dy)


async def _do_upload(page: Page, loc, file_path: str):
    """Upload a file. If the element is a real <input type=file>, set it directly;
    otherwise it's a button that opens the OS file chooser — intercept that."""
    try:
        tag = (await loc.evaluate("el => el.tagName.toLowerCase() + '|' + (el.type||'')")).split("|")
    except PWError:
        tag = ["", ""]
    if tag[0] == "input" and tag[1] == "file":
        await loc.set_input_files(file_path)
        return
    async with page.expect_file_chooser() as fc_info:
        await loc.click()
    chooser = await fc_info.value
    await chooser.set_files(file_path)


async def _do_drag(page: Page, src_id: int, dst_id: int, session_id: str = None, tab_id: str = None):
    src = await _locator(page, src_id, session_id, tab_id)
    dst = await _locator(page, dst_id, session_id, tab_id)
    sc = await _center(page, src)
    dc = await _center(page, dst)
    if not sc or not dc:
        raise HTTPException(status_code=500, detail="Could not locate drag source/target geometry.")
    await _human_move_to(page, sc["x"], sc["y"])
    await page.mouse.down()
    # Move in steps so drag-aware UIs register the motion.
    steps = 20
    for i in range(1, steps + 1):
        await page.mouse.move(sc["x"] + (dc["x"] - sc["x"]) * i / steps,
                              sc["y"] + (dc["y"] - sc["y"]) * i / steps)
        await asyncio.sleep(0.01)
    await page.mouse.up()


async def _do_wait(page: Page, req: InteractRequest):
    if req.text:
        try:
            await page.get_by_text(req.text, exact=False).first.wait_for(timeout=(req.duration or 10000))
        except PWError:
            pass
    else:
        await asyncio.sleep((req.duration or 2000) / 1000)


@app.post("/observe")
async def observe(req: TabRequest):
    await _ensure_browser()
    page = await _get_page(req.session_id, req.tab_id)
    return await _build_response(page, req.session_id, req.tab_id)


@app.post("/switch_tab")
async def switch_tab(req: TabRequest):
    await _ensure_browser()
    page = None
    if _key(req.session_id, req.tab_id) in pages:
        page = pages[_key(req.session_id, req.tab_id)]
    elif _key("_popups", req.tab_id) in pages:
        # Adopt a captured popup into the caller's session so subsequent
        # browser_interact/browser_read on this tab_id resolve to it.
        page = pages.pop(_key("_popups", req.tab_id))
        pages[_key(req.session_id, req.tab_id)] = page
    if page is None:
        raise HTTPException(status_code=404, detail=f"Tab '{req.tab_id}' not found.")
    await page.bring_to_front()
    return await _build_response(page, req.session_id, req.tab_id)


@app.post("/close_tab")
async def close_tab(req: TabRequest):
    k = _key(req.session_id, req.tab_id)
    pk = _key("_popups", req.tab_id)
    target = k if k in pages else (pk if pk in pages else None)
    if target:
        try:
            await pages[target].close()
        except PWError:
            pass
        del pages[target]
        last_elements.pop(target, None)
    return {"status": "success", "remaining_tabs": len(_tabs_for(req.session_id))}


@app.on_event("shutdown")
async def on_shutdown():
    global context, _playwright
    try:
        if context:
            await context.close()
        if _playwright:
            await _playwright.stop()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8030")))
