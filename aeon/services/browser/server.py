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
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# Patchright is a drop-in, API-compatible Playwright fork that PATCHES the
# automation tells real bot-detection probes for: it avoids the CDP
# `Runtime.enable` leak, removes the console/runtime fingerprints, and does not
# inject detectable evasion shims. Combined with real headed Chrome + a
# persistent profile + human-like input, this is about as close to a real human
# at a real browser as we can get. (Import name mirrors playwright exactly.)
from patchright.async_api import async_playwright, Page, BrowserContext, Error as PWError

# Pure-math human-motion trajectories (curved eased mouse paths, wheel-notch
# scrolls, keystroke cadence). Kept in a Playwright-free module so it is unit
# testable outside the container; imported by bare name because the container
# runs `uvicorn server:app` from /app where both files sit side by side.
from human_motion import mouse_path, scroll_ticks, type_delays, idle_drift_target
from browser_util import (
    bearer_is_authorized, is_destructive_dialog, parse_proxy, primary_locale,
    read_auth_token, valid_timezone,
)

app = FastAPI()
BROWSER_API_VERSION = "human_v6"

# --- Configuration ----------------------------------------------------------
BROWSER_AUTH_TOKEN_FILE = os.environ.get(
    "AEON_BROWSER_TOKEN_FILE", "/run/secrets/aeon_browser_token"
)
# Load this before accepting any request. A missing or insecure secret prevents
# uvicorn from starting instead of silently exposing the persistent browser.
BROWSER_AUTH_TOKEN = read_auth_token(BROWSER_AUTH_TOKEN_FILE)


@app.middleware("http")
async def require_browser_login(request: Request, call_next):
    """Require the private login token for every endpoint, including health."""
    if not bearer_is_authorized(
        request.headers.get("authorization", ""), BROWSER_AUTH_TOKEN
    ):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "detail": "Browser API login required"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# Each agent gets its OWN browser context (own cookie jar, storage, history) so
# parallel agents are independent identities instead of colliding in one profile.
# A profile is a directory under PROFILE_ROOT; "default" (== the legacy
# /profiles/default) is the persistent, shared, cross-run profile the principal
# uses so logins survive; sub-agents get their own isolated profiles.
PROFILE_ROOT = os.environ.get("AEON_BROWSER_PROFILE_ROOT", "/profiles")
DEFAULT_PROFILE = "default"
TIMEZONE = os.environ.get("AEON_BROWSER_TZ", "America/Los_Angeles")
LOCALE = os.environ.get("AEON_BROWSER_LOCALE", "en-US")
MAX_INDEXED_ELEMENTS = 250
MAX_FRAMES = 30               # cap frames scanned per observation
ACTION_SETTLE_CAP_MS = 600    # max wait for the DOM to quiesce after an action
DOWNLOAD_DIR = os.environ.get("AEON_BROWSER_DOWNLOADS", "/profiles/downloads")
SCREEN_W, SCREEN_H = 1920, 1080  # Xvfb screen (matches entrypoint.sh)

# Structural/semantic roles kept in the TEXT element list but NOT drawn as boxes
# on the screenshot — marking every table cell/heading clutters the image and
# occludes the very text the model needs to read. Interactive roles + rows/items
# (common click targets) are still marked.
_STRUCTURAL_MARK_ROLES = {"heading", "article", "cell", "gridcell"}

_geo_cache = None


def _resolve_geo():
    """(timezone_id, locale, geolocation|None) consistent with the EGRESS IP, so
    the browser's clock, language and coordinates match where it appears to
    connect from — an IP-vs-timezone/language mismatch is a top bot-detection
    signal. Explicit AEON_BROWSER_TZ / AEON_BROWSER_LOCALE env vars win; a lookup
    failure falls back to the configured defaults. Best-effort and cached."""
    global _geo_cache
    if _geo_cache is not None:
        return _geo_cache
    tz = os.environ.get("AEON_BROWSER_TZ")
    loc = os.environ.get("AEON_BROWSER_LOCALE")
    geoloc = None
    if not (tz and loc):
        # Route lookup through the SAME proxy as Chrome. ipapi routinely rate
        # limits shared/cloud egress, so use two independent providers before
        # falling back. No IP/address is logged or returned to the agent.
        proxy_raw = (os.environ.get("AEON_BROWSER_PROXY") or "").strip()
        opener = None
        if proxy_raw:
            from urllib.request import ProxyHandler, build_opener
            opener = build_opener(ProxyHandler({"http": proxy_raw, "https": proxy_raw}))
        country_locales = {
            "US": "en-US", "GB": "en-GB", "CA": "en-CA", "AU": "en-AU",
            "NZ": "en-NZ", "DE": "de-DE", "FR": "fr-FR", "ES": "es-ES",
            "IT": "it-IT", "PT": "pt-PT", "BR": "pt-BR", "MX": "es-MX",
            "JP": "ja-JP", "KR": "ko-KR", "CN": "zh-CN", "TW": "zh-TW",
            "HK": "zh-HK", "IN": "en-IN", "NL": "nl-NL", "PL": "pl-PL",
            "TR": "tr-TR", "SE": "sv-SE", "NO": "no-NO", "DK": "da-DK",
            "FI": "fi-FI",
        }
        failures = []
        for provider, endpoint in (
            ("ipwho", "https://ipwho.is/"),
            ("ipapi", "https://ipapi.co/json/"),
        ):
            try:
                req = URLRequest(endpoint, headers={"User-Agent": "Mozilla/5.0"})
                response = opener.open(req, timeout=5) if opener else urlopen(req, timeout=5)
                with response as r:
                    d = json.loads(r.read().decode())
                if provider == "ipwho":
                    if d.get("success") is False:
                        raise ValueError(str(d.get("message") or "provider rejected lookup"))
                    zone = (d.get("timezone") or {}).get("id")
                    languages = ""
                    country = str(d.get("country_code") or "").upper()
                else:
                    zone = d.get("timezone")
                    languages = d.get("languages", "")
                    country = str(d.get("country_code") or "").upper()
                tz = tz or zone
                loc = loc or primary_locale(languages) or country_locales.get(country)
                lat, lon = d.get("latitude"), d.get("longitude")
                if lat is not None and lon is not None:
                    geoloc = {
                        "latitude": float(lat), "longitude": float(lon), "accuracy": 50.0
                    }
                break
            except Exception as exc:
                failures.append(f"{provider}:{type(exc).__name__}")
        if not (tz or loc or geoloc):
            print(f"[browser] geo lookup failed ({', '.join(failures)}); using configured defaults.")
    tz = tz or TIMEZONE
    loc = loc or LOCALE
    # A malformed timezone from the lookup would make BOTH the Chrome and the
    # Chromium-fallback launch throw and brick the browser — validate it against
    # the tz database and fall back to the default if it isn't a real zone.
    if not valid_timezone(tz):
        print(f"[browser] timezone '{tz}' not recognized; using {TIMEZONE}.")
        tz = TIMEZONE
    _geo_cache = (tz, loc, geoloc)
    print(f"[browser] identity -> tz={tz} locale={loc} geo={'match-IP' if geoloc else 'none'}")
    return _geo_cache


def _resolve_proxy():
    """Playwright proxy dict from AEON_BROWSER_PROXY (e.g. 'http://user:pass@host:port'
    or 'socks5://host:port'), or None — lets the browser appear to originate from
    anywhere and avoids burning the host IP. Parsing logic lives in browser_util."""
    return parse_proxy(os.environ.get("AEON_BROWSER_PROXY") or "")

# --- Global state -----------------------------------------------------------
_playwright = None
# profile -> BrowserContext (one isolated Chrome per profile).
contexts: Dict[str, BrowserContext] = {}
# Per-profile launch locks so two concurrent requests for the SAME profile can't
# both launch Chrome on its user-data-dir (which would fail on the dir lock).
_launch_locks: Dict[str, asyncio.Lock] = {}
# page_key -> Page  (page_key = "<profile>::<session_id>::<tab_id>")
pages: Dict[str, Page] = {}
# page_key -> {aeon_id: element_info}  (last observed snapshot, for verification)
last_elements: Dict[str, Dict[int, Dict[str, Any]]] = {}
# page_key -> {aeon_id: Frame}  (which frame owns each id, for frame-aware actions)
frame_maps: Dict[str, Dict[int, Any]] = {}
# id(page) -> list of recent dialog/download notes to surface in the next obs
page_events: Dict[int, List[str]] = {}
# id(page) -> one-shot native-dialog policy for the current action. Policies
# expire quickly so an action that did not open a dialog cannot affect a later
# unrelated prompt.
dialog_policies: Dict[int, Dict[str, Any]] = {}
_popup_counter = 0
# Popup page-keys already announced to the agent, so each new tab is surfaced once.
_announced_popups: set = set()
# Best-effort tracking of the real cursor position (Playwright does not expose
# it) so each move can start a natural curved path from where the pointer
# actually is, not teleport. Seeded near the middle of the 1920x1080 Xvfb screen;
# it is self-correcting — the first real move overwrites it with the true target.
_cursor: Dict[str, float] = {"x": 960.0, "y": 540.0}


def _safe_profile(profile: Optional[str]) -> str:
    """Sanitize a profile name into a safe directory/key token."""
    p = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in (profile or DEFAULT_PROFILE))
    p = p.strip("-.") or DEFAULT_PROFILE
    return p[:64]


def _profile_dir(profile: str) -> str:
    return os.path.join(PROFILE_ROOT, _safe_profile(profile))


def _launch_lock_for(profile: str) -> asyncio.Lock:
    lock = _launch_locks.get(profile)
    if lock is None:
        lock = _launch_locks[profile] = asyncio.Lock()
    return lock


def _key(profile: str, session_id: str, tab_id: str) -> str:
    return f"{profile}::{session_id}::{tab_id}"


def _tabs_for(profile: str, session_id: str) -> List[str]:
    prefix = f"{profile}::{session_id}::"
    return [k[len(prefix):] for k in pages if k.startswith(prefix)]


# ============================================================================
# Injected JavaScript: element extraction + Set-of-Mark overlay
# ============================================================================

# Builds the stable element index. Stamps data-aeon-id on each visible
# interactable/meaningful node and returns role/name/value/state/geometry.
_EXTRACT_JS = r"""
({maxEls, startId, searchText, roleFilter}) => {
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
  const SEARCHABLE_TEXT = 'p,li,dt,dd,th,td,h1,h2,h3,h4,h5,h6,pre,code,blockquote';
  const query = (searchText || '').trim().toLowerCase();
  const wantedRole = (roleFilter || '').trim().toLowerCase();

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
  // Non-actionable visual regions used only to make lossless 2x crops for the
  // multimodal model.  They deliberately do NOT receive action ids: the normal
  // indexed element list remains the sole interaction contract.
  const regions = [];
  const regionKeys = new Set();
  function consider(el) {
    if (seen.has(el)) return;
    seen.add(el);
    let ok = false;
    try {
      ok = el.matches && (el.matches(MATCH) || (query && el.matches(SEARCHABLE_TEXT)));
    } catch (e) { ok = false; }
    if (ok && query) {
      const hay = (accName(el) + ' ' + (el.value || '') + ' ' +
                   (el.getAttribute('title') || '') + ' ' +
                   (el.getAttribute('alt') || '')).toLowerCase();
      if (!hay.includes(query)) ok = false;
    }
    if (ok && wantedRole) {
      const actualRole = (el.getAttribute('role') || implicitRole(el)).toLowerCase();
      if (actualRole !== wantedRole) ok = false;
    }
    if (ok) {
      const s = window.getComputedStyle(el);
      if (!(s.visibility === 'hidden' || s.display === 'none' || parseFloat(s.opacity || '1') === 0)) {
        const r = el.getBoundingClientRect();
        if (r.width >= 2 && r.height >= 2) cands.push({ el: el, r: r });
      }
    }
    if (regions.length < 24) {
      try {
        const tag = (el.tagName || '').toLowerCase();
        const role = (el.getAttribute('role') || '').toLowerCase();
        const ident = [el.id, el.className, el.getAttribute('aria-label'),
                       el.getAttribute('title'), el.getAttribute('src')]
          .map(x => String(x || '')).join(' ').toLowerCase();
        let kind = '';
        if (/captcha|recaptcha|hcaptcha|turnstile|verify.you.are.human|security.challenge/.test(ident)) {
          kind = 'verification';
        } else if (role === 'alert' || el.getAttribute('aria-invalid') === 'true' ||
                   /(^|[ _-])(error|invalid|warning)([ _-]|$)/.test(ident)) {
          kind = 'error';
        } else if (tag === 'table' || role === 'table' || role === 'grid') {
          kind = 'table';
        } else if (tag === 'canvas' || tag === 'svg' ||
                   ((tag === 'img' || role === 'img') && /chart|diagram|graph|plot|map/.test(ident))) {
          kind = 'diagram';
        }
        if (kind) {
          const s = window.getComputedStyle(el);
          const r = el.getBoundingClientRect();
          const visible = !(s.visibility === 'hidden' || s.display === 'none' ||
                            parseFloat(s.opacity || '1') === 0);
          const intersects = r.bottom > 0 && r.right > 0 &&
                             r.top < window.innerHeight && r.left < window.innerWidth;
          const minW = kind === 'error' ? 20 : 60;
          const minH = kind === 'error' ? 10 : 30;
          const key = kind + ':' + Math.round(r.x / 8) + ':' + Math.round(r.y / 8) + ':' +
                      Math.round(r.width / 8) + ':' + Math.round(r.height / 8);
          if (visible && intersects && r.width >= minW && r.height >= minH && !regionKeys.has(key)) {
            regionKeys.add(key);
            let label = accName(el).replace(/\s+/g, ' ').trim();
            if (label.length > 160) label = label.slice(0, 160) + '…';
            regions.push({
              kind: kind,
              label: label,
              rect: {x: Math.round(r.x), y: Math.round(r.y),
                     w: Math.round(r.width), h: Math.round(r.height)}
            });
          }
        }
      } catch (e) {}
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
    visualRegions: regions,
    nextId: startId + n,
    page: {
      scrollY: Math.round(window.scrollY),
      scrollHeight: Math.round(de.scrollHeight),
      clientHeight: Math.round(de.clientHeight || window.innerHeight),
      scrollX: Math.round(window.scrollX),
      scrollWidth: Math.round(de.scrollWidth),
      clientWidth: Math.round(de.clientWidth || window.innerWidth),
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

# Form-validation scrape. When a Submit click does nothing, the reason is almost
# always here: a required field is empty/unset or a field is flagged invalid, and
# the message is small red text a weak vision model can't read. This returns that
# state as TEXT so the agent knows exactly what to fix. Covers native HTML5
# constraint validation, aria-invalid + its described error text, visible
# alert/error nodes, and unselected <select>s (the birthday/gender dropdown case).
_VALIDATION_JS = r"""
() => {
  const vis = (el) => {
    const s = window.getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none' || parseFloat(s.opacity||'1') === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width >= 1 && r.height >= 1;
  };
  const labelFor = (el) => {
    let t = '';
    if (el.labels && el.labels.length) t = el.labels[0].innerText;
    t = t || el.getAttribute('aria-label') || el.getAttribute('placeholder')
          || el.getAttribute('name') || el.id || el.getAttribute('role') || el.tagName.toLowerCase();
    return (t || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  };
  const invalid = [];
  const seen = new Set();
  const push = (label, reason) => {
    const key = label + '|' + reason;
    if (label && !seen.has(key)) { seen.add(key); invalid.push({ label, reason }); }
  };
  const controls = Array.from(document.querySelectorAll('input, select, textarea, [contenteditable=""], [contenteditable="true"]'));
  for (const el of controls) {
    if (!vis(el) || el.disabled || el.type === 'hidden') continue;
    // Native HTML5 constraint validation (read-only; does not fire events).
    if (typeof el.checkValidity === 'function' && el.validity && el.validity.valid === false) {
      push(labelFor(el), el.validationMessage || 'invalid');
      continue;
    }
    // Explicitly flagged invalid by the site's own (JS) validation.
    if (el.getAttribute('aria-invalid') === 'true') {
      let msg = 'flagged invalid';
      const d = el.getAttribute('aria-describedby');
      if (d) {
        const parts = d.split(/\s+/).map(id => { const n = document.getElementById(id); return n ? n.innerText : ''; });
        const txt = parts.join(' ').replace(/\s+/g, ' ').trim();
        if (txt) msg = txt.slice(0, 120);
      }
      push(labelFor(el), msg);
      continue;
    }
    // Unselected dropdown — the classic silent Submit-blocker (birthday/gender).
    // Key on an empty VALUE (the placeholder option), so a select showing a real
    // default (e.g. a country already chosen) is not falsely flagged.
    if (el.tagName === 'SELECT' && el.value === '') {
      push(labelFor(el), 'no option selected');
      continue;
    }
    // Required-but-empty text field. Read text from the right place: a
    // contenteditable holds its text in innerText, not .value.
    const req = el.required || el.getAttribute('aria-required') === 'true';
    const empty = el.isContentEditable
      ? !((el.innerText || '').trim())
      : !(el.value && String(el.value).trim());
    if (req && empty) push(labelFor(el), 'required, empty');
  }
  // Visible alert/error banners (site-rendered validation messages).
  const alerts = [];
  const aseen = new Set();
  const nodes = Array.from(document.querySelectorAll('[role=alert], [aria-live=assertive]'));
  for (const n of nodes) {
    if (!vis(n)) continue;
    const t = (n.innerText || '').replace(/\s+/g, ' ').trim();
    if (t && t.length <= 200 && !aseen.has(t)) { aseen.add(t); alerts.push(t); }
    if (alerts.length >= 8) break;
  }
  return { invalid: invalid.slice(0, 20), alerts };
}
"""

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

# Compact grounded text for every normal observation. Unlike read_text (a full
# article/document extraction), this is limited to meaningful blocks intersecting
# the current viewport, so the agent can notice status banners, modal text, and
# non-interactive results without paying for an entire page body every turn.
_VISIBLE_TEXT_JS = r"""
() => {
  const selectors = [
    'h1','h2','h3','h4','h5','h6','p','li','dt','dd','th','td','caption','legend',
    '[role=alert]','[role=status]','[role=dialog]','[aria-live]','main','article'
  ].join(',');
  const out = [], seen = new Set();
  for (const el of document.querySelectorAll(selectors)) {
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none' || parseFloat(s.opacity || '1') === 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1 || r.bottom <= 0 || r.right <= 0 || r.top >= innerHeight || r.left >= innerWidth) continue;
    let text = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (!text || text.length > 800 || seen.has(text)) continue;
    seen.add(text); out.push(text);
    if (out.join('\n').length >= 5000) break;
  }
  return out.join('\n').slice(0, 5000);
}
"""


async def _visible_text(page: Page) -> str:
    chunks = []
    seen = set()
    total = 0
    for frame in page.frames[:12]:
        try:
            value = await frame.evaluate(_VISIBLE_TEXT_JS)
        except Exception:
            continue
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        room = 6000 - total
        if room <= 0:
            break
        chunks.append(value[:room])
        total += min(len(value), room)
    return "\n\n".join(chunks)[:6000]


# ============================================================================
# Browser lifecycle
# ============================================================================

async def _ensure_browser(profile: str = DEFAULT_PROFILE):
    """Return the (lazily launched) browser context for `profile`.

    Each profile is its OWN isolated Chrome (own cookie jar/storage/history), so
    agents on different profiles are independent identities. Maximum realism per
    context, in order of importance:
      1. REAL Google Chrome (channel='chrome') — correct branding, build flags,
         fonts, codecs and TLS/JA3 fingerprint, not open-source Chromium.
      2. HEADED under Xvfb — headless has dozens of detectable differences.
      3. Patchright's patched driver — no CDP `Runtime.enable` leak; injects NO
         detectable evasion shim, so we add no navigator.webdriver hacks either.
      4. NO user_agent / viewport override — the real browser is self-consistent.
      5. Persistent profile dir — real cookies/history/login state per identity.
    """
    profile = _safe_profile(profile)
    ctx = contexts.get(profile)
    if ctx is not None:
        return ctx
    # Only one coroutine launches THIS profile; others wait and reuse it.
    async with _launch_lock_for(profile):
        ctx = contexts.get(profile)
        if ctx is not None:
            return ctx
        global _playwright
        pdir = _profile_dir(profile)
        os.makedirs(pdir, exist_ok=True)
        # Clear a STALE Chrome singleton lock left by a previous container that was
        # killed without a clean shutdown. Without this, a persistent profile
        # refuses to launch after a container restart ("profile appears to be in
        # use ... on another computer", since the new container has a new hostname).
        # Safe: our per-profile launch lock guarantees no live Chrome is using this
        # dir in THIS process, and only one browser container runs at a time.
        for lock in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            try:
                os.remove(os.path.join(pdir, lock))
            except OSError:
                pass
        if _playwright is None:
            _playwright = await async_playwright().start()

        # The geo lookup does a BLOCKING http call; run it off the event loop so
        # it can't stall other requests (health checks, other tabs) for seconds.
        loop = asyncio.get_event_loop()
        tz, loc, geoloc = await loop.run_in_executor(None, _resolve_geo)
        proxy = _resolve_proxy()
        # --no-sandbox is required to run Chrome as root in a container; the rest aid Xvfb.
        # --disable-blink-features=AutomationControlled is REQUIRED for real Chrome
        # (channel="chrome") driven over CDP: without it Chrome exposes
        # navigator.webdriver=true (bot.sannysoft "WebDriver (New): present (failed)").
        # It's a launch flag — invisible to page JS, no detectable shim — that makes
        # navigator.webdriver read false exactly like a normal browser.
        # Openbox runs over Xvfb (entrypoint.sh), so --start-maximized produces a
        # real, internally consistent 1920x1080 desktop instead of Chrome's small
        # no-window-manager default (~945x973).
        args = ["--no-sandbox", "--disable-dev-shm-usage", "--start-maximized",
                "--disable-blink-features=AutomationControlled"]
        # GPU-accelerated WebGL (only when the container has a GPU, signalled by
        # AEON_BROWSER_GPU). Without this, Chrome under Xvfb falls back to the
        # SwiftShader software renderer — a datacenter/headless fingerprint tell
        # ("WebGL Renderer: SwiftShader"). With a real GPU + these flags it reports
        # the actual NVIDIA renderer like a normal desktop. Default (unset) keeps
        # the safe software path so machines without a browser GPU still work.
        gpu_flags = []
        if os.environ.get("AEON_BROWSER_GPU"):
            gpu_flags = ["--ignore-gpu-blocklist", "--enable-gpu-rasterization",
                         "--use-gl=angle", "--use-angle=vulkan", "--enable-features=Vulkan"]
            args += gpu_flags
        if proxy:
            # Behind a proxy, stop WebRTC from revealing the real local/host IP via
            # STUN — a classic leak that unmasks an otherwise-clean proxied session.
            args.append("--force-webrtc-ip-handling-policy=disable_non_proxied_udp")

        launch_kwargs = dict(
            user_data_dir=pdir,
            headless=False,
            no_viewport=True,           # use the real OS window size, no viewport tell
            accept_downloads=True,      # so the download handler can save files
            locale=loc,                 # language/Accept-Language matched to the egress IP
            timezone_id=tz,             # clock matched to the egress IP
            args=args,
        )
        if geoloc:
            # Coordinates matched to the IP, and permission pre-granted so location-
            # aware sites work smoothly and consistently (no mismatch, no prompt hang).
            launch_kwargs["geolocation"] = geoloc
            launch_kwargs["permissions"] = ["geolocation"]
        if gpu_flags:
            # Drop Playwright's SwiftShader-forcing defaults so our real-GPU flags win.
            launch_kwargs["ignore_default_args"] = ["--enable-unsafe-swiftshader", "--disable-gpu"]
        if proxy:
            launch_kwargs["proxy"] = proxy
        try:
            ctx = await _playwright.chromium.launch_persistent_context(channel="chrome", **launch_kwargs)
        except Exception as e:
            # Fall back to the patched Chromium if real Chrome isn't present.
            print(f"[browser] real Chrome channel unavailable ({e}); using patched Chromium.")
            ctx = await _playwright.chromium.launch_persistent_context(**launch_kwargs)
        # Capture popups/new tabs (OAuth windows, target=_blank) for THIS profile.
        ctx.on("page", lambda p, prof=profile: _register_popup(prof, p))
        # Crash recovery: if the context dies, drop its state so the next request
        # for this profile transparently relaunches instead of wedging.
        ctx.on("close", lambda prof=profile: _on_context_close(prof))
        contexts[profile] = ctx
        print(f"[browser] launched context for profile '{profile}' ({pdir})")
    return ctx


def _purge_profile_state(profile: str):
    """Drop all in-memory page/frame/event state for a profile's keys."""
    prefix = f"{profile}::"
    for k in [k for k in pages if k.startswith(prefix)]:
        pg = pages.pop(k, None)
        last_elements.pop(k, None)
        frame_maps.pop(k, None)
        _announced_popups.discard(k)
        if pg is not None:
            page_events.pop(id(pg), None)


def _on_context_close(profile: str):
    contexts.pop(profile, None)
    _purge_profile_state(profile)


def _note_event(page: Page, msg: str):
    """Record bounded, de-duplicated browser-engine evidence for the next turn."""
    clean = str(msg or "").replace("\n", " ").strip()[:500]
    if not clean:
        return
    events = page_events.setdefault(id(page), [])
    if clean not in events[-12:]:
        events.append(clean)
    del events[:-40]


def _event_url(raw: str) -> str:
    """Remove query/fragment credentials from a diagnostic URL."""
    try:
        parts = urlsplit(str(raw or ""))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))[:240]
    except Exception:
        return "(unknown URL)"


def _on_request_failed(page: Page, request):
    try:
        if request.resource_type not in {"document", "xhr", "fetch"}:
            return
        failure = request.failure or "request failed"
        _note_event(
            page,
            f"[network:{request.resource_type}] {failure} — {_event_url(request.url)}",
        )
    except Exception:
        pass


def _on_response(page: Page, response):
    try:
        kind = response.request.resource_type
        if response.status < 400 or kind not in {"document", "xhr", "fetch"}:
            return
        _note_event(
            page,
            f"[http:{response.status}:{kind}] {_event_url(response.url)}",
        )
    except Exception:
        pass


def _on_console(page: Page, message):
    try:
        level = str(message.type or "").lower()
        if level in {"error", "warning"}:
            _note_event(page, f"[console:{level}] {str(message.text or '')[:300]}")
    except Exception:
        pass


def _setup_page(page: Page):
    """Attach dialog / download / crash handlers to a page we manage."""
    page.on("dialog", lambda d: asyncio.create_task(_handle_dialog(page, d)))
    page.on("download", lambda d: asyncio.create_task(_handle_download(page, d)))
    page.on("crash", lambda: _note_event(page, "[page crashed — reload or re-navigate]"))
    page.on("requestfailed", lambda request: _on_request_failed(page, request))
    page.on("response", lambda response: _on_response(page, response))
    page.on("console", lambda message: _on_console(page, message))
    page.on("pageerror", lambda error: _note_event(page, f"[page JS error] {str(error)[:300]}"))


async def _handle_dialog(page: Page, dialog):
    # Native alert/confirm/prompt/beforeunload would otherwise block the next
    # action. Default is to accept (so flows proceed) and report it — EXCEPT a
    # confirm/prompt whose text looks irreversible (delete/discard/overwrite...),
    # which we DISMISS so the agent never silently triggers a destructive action.
    # It is told, and can redo it deliberately if that was actually the intent.
    dtype = dialog.type
    msg = (dialog.message or "")[:200]
    policy = dialog_policies.pop(id(page), None) or {}
    if float(policy.get("expires", 0)) < time.monotonic():
        policy = {}
    requested = str(policy.get("action") or "auto").lower()
    if requested == "dismiss":
        _note_event(page, f"[dialog:{dtype}] {msg} -> DISMISSED (explicit response)")
        try:
            await dialog.dismiss()
        except Exception:
            pass
        return
    if requested == "accept":
        prompt_text = str(policy.get("text") or "")
        _note_event(
            page,
            f"[dialog:{dtype}] {msg} -> ACCEPTED (explicit response"
            + (" with supplied prompt text)" if dtype == "prompt" else ")"),
        )
        try:
            await dialog.accept(prompt_text if dtype == "prompt" else "")
        except Exception:
            try:
                await dialog.dismiss()
            except Exception:
                pass
        return
    if dtype in ("confirm", "prompt") and is_destructive_dialog(msg):
        _note_event(page, f"[dialog:{dtype}] {msg} -> DISMISSED (looked destructive; NOT auto-confirmed). "
                          f"If intended, repeat with dialog_action='accept'.")
        try:
            await dialog.dismiss()
        except Exception:
            pass
        return
    # A prompt needs intentional text. Dismissing it is more truthful than
    # silently submitting an empty value; the agent can repeat with dialog_text.
    if dtype == "prompt":
        _note_event(
            page,
            f"[dialog:prompt] {msg} -> DISMISSED (no dialog_text supplied; repeat with "
            f"dialog_action='accept' and dialog_text='...')",
        )
        try:
            await dialog.dismiss()
        except Exception:
            pass
        return
    _note_event(page, f"[dialog:{dtype}] {msg} (auto-accepted)")
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


def _register_popup(profile: str, page: Page):
    # The "page" event also fires for pages WE create via new_page(); defer to an
    # async task that checks opener() so we only capture genuine popups (OAuth /
    # target=_blank windows), never our own tabs, and avoid a registration race.
    asyncio.create_task(_maybe_register_popup(profile, page))


async def _maybe_register_popup(profile: str, page: Page):
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
    # Popups belong to the SAME profile/context that spawned them.
    pages[_key(profile, "_popups", f"popup_{_popup_counter}")] = page


async def _get_page(profile: str, session_id: str, tab_id: str) -> Page:
    k = _key(profile, session_id, tab_id)
    page = pages.get(k)
    if page is None:
        raise HTTPException(status_code=404, detail=f"Tab '{tab_id}' not found. Navigate to open it.")
    if page.is_closed():
        # A tab that closed itself (e.g. an OAuth popup) — clean it up and report.
        pages.pop(k, None); last_elements.pop(k, None); frame_maps.pop(k, None)
        raise HTTPException(status_code=404,
                            detail=f"Tab '{tab_id}' has closed. Switch to another open tab or navigate to re-open it.")
    return page


async def _reconcile_pages(profile: str, page: Page):
    """Sync our page registry with the context's REAL pages so tab/popup handling
    is reliable regardless of async-event timing:
      * capture any tab the SITE opened (popup / target=_blank / window.open) that
        we aren't tracking yet — so it shows up in open_tabs THIS observation,
      * drop any tracked page that has CLOSED (e.g. an OAuth popup that finished),
        so we never hand back a dead tab.
    Idempotent with the async popup handler (whichever registers a page first wins;
    the other sees it and skips)."""
    global _popup_counter
    try:
        ctx_pages = list(page.context.pages)
    except Exception:
        return
    prefix = f"{profile}::"
    # Drop closed pages we still track for this profile.
    for k, p in [(k, p) for k, p in pages.items() if k.startswith(prefix)]:
        try:
            if p.is_closed():
                pages.pop(k, None)
                last_elements.pop(k, None)
                frame_maps.pop(k, None)
                page_events.pop(id(p), None)
                _announced_popups.discard(k)
        except Exception:
            pass
    # Capture any untracked, still-open page in this context as a popup tab.
    known = set(pages.values())
    for p in ctx_pages:
        try:
            if p in known or p.is_closed():
                continue
        except Exception:
            continue
        # Skip the context's INITIAL about:blank page — a persistent context opens
        # with one, and it is NOT a popup. A real popup has an opener (the page
        # that spawned it) even while it is momentarily about:blank; the initial
        # page has none. Only skip the blank+opener-less case.
        try:
            is_blank = p.url in ("about:blank", "")
        except Exception:
            is_blank = False
        if is_blank:
            try:
                opener = await p.opener()
            except Exception:
                opener = None
            if opener is None:
                continue
        _setup_page(p)
        _popup_counter += 1
        pages[_key(profile, "_popups", f"popup_{_popup_counter}")] = p


def _announce_new_tabs(profile: str, page: Page):
    """Surface, ONCE, every popup tab the agent hasn't been told about yet — on the
    current page's events. Decoupled from capture so a popup is announced whether
    the async 'page' handler or the reconciliation loop grabbed it first."""
    prefix = f"{profile}::_popups::"
    for k in [k for k in pages if k.startswith(prefix)]:
        if k in _announced_popups:
            continue
        _announced_popups.add(k)
        tab_name = k[len(prefix):]
        try:
            purl = pages[k].url or "(loading)"
        except Exception:
            purl = "(unknown)"
        _note_event(page, f"[tab] a new tab '{tab_name}' opened ({purl[:80]}); "
                          f"browser_switch_tab(tab_id='{tab_name}') to view/use it, or close it.")


# ============================================================================
# Human-like input helpers
# ============================================================================

async def _human_pause(lo=0.05, hi=0.18):
    await asyncio.sleep(random.uniform(lo, hi))


async def _human_move_to(page: Page, x: float, y: float):
    """Move the cursor to (x, y) along a CURVED, eased, jittered path in real
    time — sampled point-by-point with short sleeps between samples so the
    trajectory has human curvature AND a human velocity profile (slow-fast-slow),
    not a single instantaneous straight-line interpolation. Tracks the cursor so
    the next move starts from the true current position."""
    start = (_cursor["x"], _cursor["y"])
    path = mouse_path(start, (x, y))
    for i, (px, py) in enumerate(path):
        await page.mouse.move(px, py)
        # Slightly longer dwell on the last couple of samples (settling on target).
        if i >= len(path) - 2:
            await asyncio.sleep(random.uniform(0.010, 0.022))
        else:
            await asyncio.sleep(random.uniform(0.005, 0.013))
    _cursor["x"], _cursor["y"] = float(x), float(y)


def _frame_for(profile: str, session_id: str, tab_id: str, aeon_id: int, page: Page):
    """The frame that owns this id (so iframe/shadow elements resolve correctly);
    falls back to the main frame."""
    return frame_maps.get(_key(profile, session_id, tab_id), {}).get(aeon_id) or page.main_frame


async def _locator(page: Page, aeon_id: int, profile: str = None, session_id: str = None, tab_id: str = None):
    owner = _frame_for(profile, session_id, tab_id, aeon_id, page) if session_id is not None else page
    return owner.locator(f'[data-aeon-id="{aeon_id}"]').first


async def _center(page: Page, locator) -> Optional[Dict[str, float]]:
    try:
        box = await locator.bounding_box()
    except PWError:
        box = None
    if not box:
        return None
    return {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}


async def _viewport_size(page: Page) -> Optional[Dict[str, float]]:
    try:
        return await page.evaluate("() => ({w: window.innerWidth, h: window.innerHeight})")
    except PWError:
        return None


async def _click_point(page: Page, locator) -> Optional[Dict[str, float]]:
    """A randomized point within the inner ~30–70% of the element's VISIBLE region
    (clamped to the viewport). Randomized because a fixed dead-center click-point
    distribution is a subtle automation tell; clamped so a large or partially
    off-screen element still gets an on-screen, on-element click. Returns None when
    no part is visible, so the caller falls back to Playwright's own click."""
    try:
        box = await locator.bounding_box()
    except PWError:
        box = None
    if not box:
        return None
    left, top = box["x"], box["y"]
    right, bottom = box["x"] + box["width"], box["y"] + box["height"]
    vp = await _viewport_size(page)
    if vp:
        left, top = max(left, 1.0), max(top, 1.0)
        right, bottom = min(right, vp["w"] - 1.0), min(bottom, vp["h"] - 1.0)
    if right <= left or bottom <= top:
        return None  # nothing visible to aim at
    return {"x": left + (right - left) * random.uniform(0.3, 0.7),
            "y": top + (bottom - top) * random.uniform(0.3, 0.7)}


async def _human_click(page: Page, locator, button: str = "left", clicks: int = 1):
    await locator.scroll_into_view_if_needed(timeout=5000)
    c = await _click_point(page, locator)
    if c:
        await _human_move_to(page, c["x"], c["y"])
        await _human_pause()
        await page.mouse.click(c["x"], c["y"], button=button, click_count=clicks,
                               delay=random.randint(40, 110))
    else:
        # Fallback to element click if geometry is unavailable.
        await locator.click(button=button, click_count=clicks, timeout=5000)


async def _element_disabled(locator) -> bool:
    """True if the element is disabled (native `disabled` or aria-disabled='true')."""
    try:
        if not await locator.is_enabled(timeout=800):
            return True
    except PWError:
        pass
    try:
        if ((await locator.get_attribute("aria-disabled")) or "").lower() == "true":
            return True
    except PWError:
        pass
    return False


async def _wait_enabled_or_note(locator, element_id) -> Optional[str]:
    """Return None if the target is (or shortly becomes) enabled, else a note saying
    it is disabled and why that matters.

    A human/coordinate click (see _human_click -> page.mouse.click) fires at a point
    and BYPASSES Playwright's built-in enabled check — so clicking a disabled control
    (e.g. a Submit/Next greyed out while an async field-validation runs) is a SILENT
    no-op. The agent then sees an unchanged page and wrongly concludes the value was
    rejected/taken, and loops. So: wait a bounded time for an async check to enable
    it; if it stays disabled, hand back a clear reason instead of a dead click."""
    for _ in range(13):  # ~2.5s max
        if not await _element_disabled(locator):
            return None
        await asyncio.sleep(0.2)
    label = ""
    try:
        label = ((await locator.get_attribute("aria-label")) or (await locator.inner_text()) or "")[:40]
    except PWError:
        pass
    tag = f" '{label.strip()}'" if label.strip() else ""
    return (f"⚠ TARGET DISABLED: element [{element_id}]{tag} is disabled, so the click did NOT "
            f"register (no effect). The form's precondition is unmet — a required field is empty "
            f"or invalid, or an availability/validation check is still pending or failed. Fix the "
            f"blocking field (see FORM VALIDATION) before clicking; re-clicking this control will "
            f"keep doing nothing.")


async def _human_field_clear(page: Page, locator):
    """Clear a focused field the way a person does: select-all, then delete —
    real keystrokes, not an instant programmatic fill. A fill() fallback only
    fires if the field somehow ignored the keyboard clear, so correctness holds."""
    await page.keyboard.press("Control+A")
    await _human_pause(0.03, 0.09)
    await page.keyboard.press("Delete")
    try:
        remaining = await locator.input_value()
        if remaining:
            await locator.fill("")   # safety net for inputs that ignore select-all
    except PWError:
        pass  # non-input (e.g. contenteditable) — the keyboard clear is what we have


async def _human_type(page: Page, locator, text: str, clear_first: bool, then_enter: bool):
    await _human_click(page, locator)
    await _human_pause()
    if clear_first:
        await _human_field_clear(page, locator)
    await _human_type_text(page, text, then_enter)


async def _human_type_text(page: Page, text: str, then_enter: bool = False):
    """Type into the currently focused target with a natural per-key cadence."""
    # Real per-key cadence with occasional inter-word hesitation (see type_delays).
    for ch, d in zip(text, type_delays(text)):
        await page.keyboard.type(ch)
        await asyncio.sleep(d)
    if then_enter:
        await _human_pause()
        await page.keyboard.press("Enter")


async def _coordinate_target(page: Page, x: Optional[float], y: Optional[float]) -> Dict[str, Any]:
    """Validate a screenshot coordinate and describe the DOM target under it."""
    if x is None or y is None:
        raise HTTPException(status_code=400, detail="coordinate action requires x and y.")
    try:
        px, py = float(x), float(y)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="x and y must be numbers.")
    vp = await _viewport_size(page)
    if not vp or not (0 <= px < vp["w"] and 0 <= py < vp["h"]):
        dims = f"{vp['w']:.0f}x{vp['h']:.0f}" if vp else "unknown"
        raise HTTPException(
            status_code=400,
            detail=f"coordinate ({px:g}, {py:g}) is outside the current viewport ({dims}).",
        )
    info = await page.evaluate(
        """({x,y}) => {
          const e = document.elementFromPoint(x,y);
          if (!e) return {tag:'none', role:'', name:''};
          const text = (e.getAttribute('aria-label') || e.getAttribute('title') ||
                        e.innerText || e.value || '').replace(/\\s+/g,' ').trim();
          return {tag:e.tagName.toLowerCase(), role:e.getAttribute('role') || '',
                  name:text.slice(0,120)};
        }""",
        {"x": px, "y": py},
    )
    return {"x": px, "y": py, **(info or {})}


async def _drag_points(page: Page, start: Dict[str, float], end: Dict[str, float]):
    await _human_move_to(page, start["x"], start["y"])
    await _human_pause()
    await page.mouse.down()
    await _human_pause(0.08, 0.16)
    path = mouse_path(
        (start["x"], start["y"]), (end["x"], end["y"]),
        jitter=0.5, overshoot=False,
    )
    for px, py in path:
        await page.mouse.move(px, py)
        await asyncio.sleep(random.uniform(0.010, 0.024))
    await _human_pause(0.06, 0.14)
    await page.mouse.up()
    _cursor["x"], _cursor["y"] = float(end["x"]), float(end["y"])


async def _human_set_checked(page: Page, locator, desired: bool):
    """Set a checkbox/switch/radio to `desired` by CLICKING it like a person
    (curved move + real click), instead of Playwright's instant check()/uncheck().
    Reads the current state (native input OR aria-checked) and only clicks when a
    change is needed. A robust Playwright fallback fires only if the human click
    didn't register the toggle, so idempotency/correctness never regress."""
    async def _state():
        try:
            return await locator.is_checked()
        except PWError:
            try:
                return (await locator.get_attribute("aria-checked")) == "true"
            except PWError:
                return None
    current = await _state()
    if current is desired:
        return  # already in the desired state — a person wouldn't click again
    await _human_click(page, locator)
    await asyncio.sleep(0.06)
    now = await _state()
    if now is not None and now is not desired:
        try:
            await (locator.check() if desired else locator.uncheck())
        except PWError:
            pass


async def _human_clear(page: Page, locator):
    """Clear a field with human motion: click into it, then select-all + delete."""
    await _human_click(page, locator)
    await _human_pause()
    await _human_field_clear(page, locator)


async def _human_idle_drift(page: Page):
    """A small idle 'reading' cursor wander via the normal curved path. Real people
    nudge the pointer while reading a page; a cursor that only moves to click is a
    behavioral tell (reCAPTCHA v3 & friends score pointer activity)."""
    tgt = idle_drift_target((_cursor["x"], _cursor["y"]), SCREEN_W, SCREEN_H, rng=random)
    await _human_move_to(page, tgt[0], tgt[1])


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


# Resolves once the DOM has been quiet (no mutations) for `quiet` ms, or after
# `cap` ms — whichever comes first. Returns fast on a stable page and stays
# patient on one still rendering, instead of a fixed sleep that is usually too
# long and occasionally too short.
_SETTLE_JS = """
({quiet, cap}) => new Promise(resolve => {
  let last = performance.now();
  let obs;
  try {
    obs = new MutationObserver(() => { last = performance.now(); });
    obs.observe(document, {subtree:true, childList:true, attributes:true, characterData:true});
  } catch (e) {}
  const start = performance.now();
  (function check(){
    const now = performance.now();
    if (now - last >= quiet || now - start >= cap) {
      if (obs) obs.disconnect();
      resolve(Math.round(now - start));
    } else {
      setTimeout(check, 25);
    }
  })();
})
"""


async def _settle_action(page: Page):
    """Adaptive settle after an in-page interaction. If the action navigated,
    domcontentloaded fires; then we wait only until the DOM stops mutating (a
    ~120ms quiet window), capped at ACTION_SETTLE_CAP_MS. This returns in tens of
    ms on a stable page yet waits out a churning SPA — no fixed-sleep tax and no
    unbounded networkidle hang."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=4000)
    except PWError:
        pass
    try:
        await page.evaluate(_SETTLE_JS, {"quiet": 120, "cap": ACTION_SETTLE_CAP_MS})
    except PWError:
        await asyncio.sleep(0.15)


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


async def _extract(page: Page, profile: str, session_id: str, tab_id: str,
                   search_text: str = "", role_filter: str = "") -> Dict[str, Any]:
    """Run the extractor in EVERY frame (main + same/cross-origin iframes), each
    piercing its own shadow DOM, and merge into one globally-numbered element list
    with absolute (main-viewport) coordinates. Tracks which frame owns each id so
    actions can target the right frame."""
    key = _key(profile, session_id, tab_id)
    elements: List[Dict[str, Any]] = []
    visual_regions: List[Dict[str, Any]] = []
    fmap: Dict[int, Any] = {}
    main_state = {"scrollY": 0, "scrollHeight": 0, "clientHeight": 0,
                  "scrollX": 0, "scrollWidth": 0, "clientWidth": 1920,
                  "innerWidth": 1920, "innerHeight": 1080}

    start = 1
    for frame in page.frames[:MAX_FRAMES]:
        if start - 1 >= MAX_INDEXED_ELEMENTS:
            break
        off = await _frame_offset(page, frame)
        if off is None:
            continue
        try:
            data = await frame.evaluate(
                _EXTRACT_JS,
                {
                    "maxEls": MAX_INDEXED_ELEMENTS - (start - 1),
                    "startId": start,
                    "searchText": search_text,
                    "roleFilter": role_filter,
                },
            )
        except Exception:
            continue
        # A frame can return an unexpected shape (JS quirk, detach mid-eval); skip
        # it rather than let one bad frame 500 the entire observation.
        if not isinstance(data, dict):
            continue
        if frame == page.main_frame and isinstance(data.get("page"), dict):
            main_state = data["page"]
        ox, oy = off["x"], off["y"]
        iw = main_state.get("innerWidth", 1920)
        ih = main_state.get("innerHeight", 1080)
        for e in data.get("elements") or []:
            if not isinstance(e, dict) or not isinstance(e.get("rect"), dict) or "id" not in e:
                continue
            r = e["rect"]
            ax, ay = r.get("x", 0) + ox, r.get("y", 0) + oy
            w, h = r.get("w", 0), r.get("h", 0)
            e["rect"] = {"x": round(ax), "y": round(ay), "w": w, "h": h}
            e["inViewport"] = (ay + h > 0 and ax + w > 0 and ay < ih and ax < iw)
            if frame != page.main_frame:
                e["inFrame"] = True
            fmap[e["id"]] = frame
            elements.append(e)
        for region in data.get("visualRegions") or []:
            if not isinstance(region, dict) or not isinstance(region.get("rect"), dict):
                continue
            r = region["rect"]
            ax, ay = r.get("x", 0) + ox, r.get("y", 0) + oy
            w, h = r.get("w", 0), r.get("h", 0)
            in_view = (ay + h > 0 and ax + w > 0 and ay < ih and ax < iw)
            if not in_view:
                continue
            visual_regions.append({
                "kind": str(region.get("kind") or "visual")[:32],
                "label": str(region.get("label") or "")[:160],
                "rect": {"x": round(ax), "y": round(ay), "w": w, "h": h},
                "inFrame": frame != page.main_frame,
            })
        start = data.get("nextId", start)

    last_elements[key] = {e["id"]: e for e in elements}
    frame_maps[key] = fmap
    return {"elements": elements, "visual_regions": visual_regions[:24], "page": main_state}


# Detects the account the browser is CURRENTLY signed in as, from the page itself.
# The agent operates a persistent, shared profile, so a "create a new account" task
# can start already logged in as someone else — and a weak model will happily report
# that stranger's inbox as "the account I just made". Surfacing the real identity as
# ground truth in every observation makes that impossible to miss. Returns a real
# email string found in an account affordance, or null — never a guess.
_IDENTITY_JS = r"""
() => {
  const emailRe = /[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}/i;
  // 1) Explicit account affordances (Google's One-Bar chip, sign-out link, etc.).
  const strong = [
    'a[aria-label^="Google Account"]', '[aria-label*="Google Account"]',
    'a[href*="SignOutOptions"]', 'a[href*="Logout"]', 'a[href*="logout"]',
  ];
  for (const s of strong) {
    for (const el of document.querySelectorAll(s)) {
      const m = ((el.getAttribute('aria-label') || '') + ' ' + (el.textContent || '')).match(emailRe);
      if (m) return m[0];
    }
  }
  // 2) Generic: an aria-label that is clearly about the signed-in account/profile.
  for (const el of document.querySelectorAll('[aria-label]')) {
    const al = el.getAttribute('aria-label') || '';
    if (/account|signed in|logged in|profile/i.test(al)) {
      const m = al.match(emailRe);
      if (m) return m[0];
    }
  }
  return null;
}
"""


async def _detect_identity(page: Page) -> Optional[str]:
    """Best-effort signed-in account for this page. Never raises, never guesses."""
    try:
        ident = await page.evaluate(_IDENTITY_JS)
        return ident if isinstance(ident, str) and "@" in ident else None
    except PWError:
        return None
    except Exception:
        return None


async def _ensure_paintable(page: Page, tries: int = 40, delay: float = 0.5):
    """Wait until the page reports a non-zero viewport before we screenshot it.

    Right after a cold container start, Xvfb/Chrome can leave the headed window at
    0x0 for several seconds; page.screenshot() then hard-fails with "Cannot take
    screenshot with 0 width" and the whole browser action errors out. This returns
    immediately once the window has real dimensions (the warm/common case) and,
    only if it never does within the budget, forces an explicit viewport as a
    last resort so we can still capture a frame instead of failing the action.
    """
    for _ in range(tries):
        try:
            sz = await page.evaluate("({w: window.innerWidth, h: window.innerHeight})")
        except Exception:
            sz = None
        if sz and sz.get("w") and sz.get("h"):
            return
        await asyncio.sleep(delay)
    try:  # best-effort fallback; may be rejected under a no_viewport context
        await page.set_viewport_size({"width": SCREEN_W, "height": SCREEN_H})
    except Exception:
        pass


async def _build_response(page: Page, profile: str, session_id: str, tab_id: str,
                          overlay: bool = True, search_text: str = "",
                          role_filter: str = "") -> Dict[str, Any]:
    # Reliably surface any tab the site just opened, and prune any that closed,
    # before we report open_tabs — so popup handling never races an async event.
    await _reconcile_pages(profile, page)
    _announce_new_tabs(profile, page)
    extracted = await _extract(
        page, profile, session_id, tab_id,
        search_text=search_text, role_filter=role_filter,
    )

    # Clean viewport screenshot (readable, real resolution) + Set-of-Mark overlay
    # drawn from the merged absolute coordinates (covers iframe/shadow elements).
    # quality=90: the agent's multimodal model reads this exact JPEG (passed
    # through without re-encoding), so keep small page text/labels legible.
    await _ensure_paintable(page)  # robust to a cold-start 0-width window
    clean_bytes = await page.screenshot(type="jpeg", quality=90)
    overlay_bytes = None
    # The numbered overlay screenshot is only needed when the caller will actually
    # look at the marks. When not requested, skip the overlay draw + second
    # screenshot + clear (three page ops) — a meaningful per-action latency cut
    # with zero loss, since the structured element index is the lossless channel.
    if overlay:
        # Mark interactive elements + rows/list items (common click targets), but
        # NOT structural roles (headings, table cells) — those stay in the text
        # list; boxing them just clutters the image and occludes page text.
        overlay_items = [
            {"id": e["id"], "x": e["rect"]["x"], "y": e["rect"]["y"], "w": e["rect"]["w"], "h": e["rect"]["h"]}
            for e in extracted["elements"]
            if e.get("inViewport") and e.get("role") not in _STRUCTURAL_MARK_ROLES
        ]
        try:
            await page.evaluate(_OVERLAY_JS, overlay_items)
            overlay_bytes = await page.screenshot(type="jpeg", quality=90)
        except PWError:
            overlay_bytes = None
        finally:
            try:
                await page.evaluate(_OVERLAY_CLEAR_JS)
            except PWError:
                pass

    title = await page.title()
    identity = await _detect_identity(page)
    visible_text = await _visible_text(page)
    # Drain any dialog/download notes captured since the last observation.
    events = page_events.pop(id(page), [])
    return {
        "status": "success",
        "session_id": session_id,
        "tab_id": tab_id,
        "url": page.url,
        "title": title,
        "identity": identity,
        "visible_text": visible_text,
        "content_hash": hashlib.sha256(visible_text.encode("utf-8")).hexdigest()[:16],
        "search": ({"text": search_text, "role": role_filter}
                   if search_text or role_filter else None),
        "clean_b64": base64.b64encode(clean_bytes).decode(),
        # Present only when the numbered overlay was actually drawn this call.
        "overlay_b64": base64.b64encode(overlay_bytes).decode() if overlay_bytes is not None else None,
        "elements": extracted["elements"],
        "visual_regions": extracted["visual_regions"],
        "page_state": extracted["page"],
        "validation": await _scrape_validation(page),
        "events": events,
        "open_tabs": _tabs_for(profile, session_id) + _tabs_for(profile, "_popups"),
    }


async def _scrape_validation(page: Page) -> Dict[str, Any]:
    """Return {'invalid': [{label, reason}], 'alerts': [str]} describing why a form
    won't submit — empty required fields, unselected dropdowns, flagged-invalid
    inputs, and visible error banners. Best-effort across the main frame and any
    same-origin iframes; never raises (a scrape failure must not break an action)."""
    invalid, alerts, seen, aseen = [], [], set(), set()
    # Bound work on ad/tracker-heavy pages: cross-origin frames throw fast (caught),
    # but same-origin ones each run the script, so cap how many we scan.
    for frame in page.frames[:12]:
        try:
            res = await frame.evaluate(_VALIDATION_JS)
        except Exception:
            continue
        if not isinstance(res, dict):
            continue
        for item in (res.get("invalid") or []):
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            key = f"{label}|{item.get('reason')}"
            if label and key not in seen:
                seen.add(key)
                invalid.append({"label": label, "reason": item.get("reason", "invalid")})
        for a in (res.get("alerts") or []):
            if isinstance(a, str) and a not in aseen:
                aseen.add(a)
                alerts.append(a)
        if len(invalid) >= 20 and len(alerts) >= 8:
            break
    return {"invalid": invalid[:20], "alerts": alerts[:8]}


# ============================================================================
# Request models
# ============================================================================

class NavigateRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    profile: str = DEFAULT_PROFILE
    url: str
    overlay: Optional[bool] = True


class TabRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    profile: str = DEFAULT_PROFILE
    overlay: Optional[bool] = True


class FindRequest(TabRequest):
    text: str
    role: Optional[str] = None


class ExtractRequest(TabRequest):
    mode: str
    max_items: int = 200


class ProfileRequest(BaseModel):
    profile: str = DEFAULT_PROFILE


class InteractRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    profile: str = DEFAULT_PROFILE
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
    x: Optional[float] = None               # screenshot-space coordinate actions
    y: Optional[float] = None
    to_x: Optional[float] = None            # drag_at destination
    to_y: Optional[float] = None
    dialog_action: Optional[str] = None      # auto | accept | dismiss
    dialog_text: Optional[str] = None        # text for a native prompt
    overlay: Optional[bool] = True          # draw+shoot the Set-of-Mark overlay (skip when no vision needed)


class CaptureRequest(BaseModel):
    session_id: str = "default"
    tab_id: str = "default"
    profile: str = DEFAULT_PROFILE
    media_id: Optional[int] = None          # which enumerated item to save; None -> just list them
    duration_s: Optional[int] = 8           # for the video screen-recording fallback


_STRUCTURED_EXTRACT_JS = r"""
({mode, maxItems}) => {
  const clean = v => String(v == null ? '' : v).replace(/\s+/g, ' ').trim();
  const roots = [document];
  const seen = new Set();
  for (let ri = 0; ri < roots.length; ri++) {
    const root = roots[ri];
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot && !seen.has(el.shadowRoot)) {
        seen.add(el.shadowRoot); roots.push(el.shadowRoot);
      }
    }
  }
  const all = selector => {
    const out = [];
    for (const root of roots) for (const el of root.querySelectorAll(selector)) out.push(el);
    return out;
  };
  const visible = el => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > 0 &&
           (r.width > 0 || r.height > 0);
  };
  const label = el => {
    const aria = clean(el.getAttribute('aria-label'));
    if (aria) return aria;
    const ids = clean(el.getAttribute('aria-labelledby'));
    if (ids) {
      const t = clean(ids.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' '));
      if (t) return t;
    }
    if (el.labels?.length) {
      const t = clean(Array.from(el.labels).map(x => x.innerText).join(' '));
      if (t) return t;
    }
    return clean(el.placeholder || el.title || el.name || el.id || '');
  };
  if (mode === 'links') {
    return all('a[href]').filter(visible).slice(0, maxItems).map(a => ({
      text: clean(a.innerText || a.getAttribute('aria-label') || a.title).slice(0, 500),
      href: a.href, target: a.target || '', rel: a.rel || ''
    }));
  }
  if (mode === 'forms') {
    const controls = all('input,select,textarea,button,[contenteditable=true],[role=textbox],[role=combobox]')
      .filter(el => visible(el) && String(el.type || '').toLowerCase() !== 'hidden')
      .slice(0, maxItems);
    return controls.map(el => {
      const type = String(el.type || el.getAttribute('role') || el.tagName).toLowerCase();
      const isSecret = type === 'password';
      const item = {
        tag: el.tagName.toLowerCase(), type, label: label(el), name: el.name || '',
        id: el.id || '', required: !!el.required || el.getAttribute('aria-required') === 'true',
        disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
        checked: typeof el.checked === 'boolean' ? el.checked : null,
        value: isSecret ? '[REDACTED]' : clean(el.value || el.innerText).slice(0, 1000),
        form: clean(el.form?.getAttribute('aria-label') || el.form?.name || el.form?.id || '')
      };
      if (el.tagName.toLowerCase() === 'select') {
        item.options = Array.from(el.options).slice(0, 200).map(o => ({
          label: clean(o.text), value: o.value, selected: o.selected, disabled: o.disabled
        }));
      }
      return item;
    });
  }
  if (mode === 'tables') {
    return all('table,[role=table],[role=grid]').filter(visible).slice(0, 50).map((table, index) => {
      const rows = Array.from(table.querySelectorAll('tr,[role=row]')).slice(0, maxItems);
      const matrix = rows.map(row => Array.from(row.querySelectorAll('th,td,[role=columnheader],[role=rowheader],[role=gridcell],[role=cell]'))
        .slice(0, 50).map(cell => clean(cell.innerText).slice(0, 1000)));
      return {index, caption: clean(table.caption?.innerText || table.getAttribute('aria-label') || ''), rows: matrix};
    });
  }
  return [];
}
"""


# ============================================================================
# Endpoints
# ============================================================================

@app.exception_handler(PWError)
async def _pwerror_handler(request: Request, exc: PWError):
    """Any Playwright error that escapes an endpoint (e.g. a page crashing during
    screenshot/extract on observe/navigate/switch) becomes a clean, readable 500
    with a 'detail' the tool surfaces — never an opaque stack trace."""
    return JSONResponse(status_code=500, content={"status": "error",
                                                  "detail": f"Browser engine error: {exc}"})


@app.get("/health")
async def health():
    return {"status": "ok", "auth_required": True, "api_version": BROWSER_API_VERSION}


@app.get("/ping")
async def ping():
    return {"status": "pong", "auth_required": True, "version": BROWSER_API_VERSION}


# Schemes that are complete as-is (no host to prepend). "javascript" is
# deliberately excluded — we never navigate to javascript: URLs.
_ABSOLUTE_SCHEMES = ("data", "about", "blob", "file", "chrome", "view-source", "ftp")


def _normalize_url(raw: str) -> str:
    """Turn user input into a navigable URL. Prepends https:// ONLY when there is
    no scheme — correctly leaving http(s)://, data:, about:, file:, etc. alone, and
    NOT mistaking a host:port ('example.com:8080') for a scheme."""
    u = (raw or "").strip()
    if "://" in u:
        return u
    head = u.split(":", 1)[0].lower() if ":" in u else ""
    if head in _ABSOLUTE_SCHEMES:
        return u
    return "https://" + u


@app.post("/navigate")
async def navigate(req: NavigateRequest):
    profile = _safe_profile(req.profile)
    ctx = await _ensure_browser(profile)
    k = _key(profile, req.session_id, req.tab_id)
    # A tab that closed itself should transparently re-open on navigate.
    if k in pages and pages[k].is_closed():
        del pages[k]
    if k not in pages:
        page = await ctx.new_page()
        _setup_page(page)
        pages[k] = page
    page = pages[k]
    url = _normalize_url(req.url)
    # One retry on a TRANSIENT network error (timeouts, resets) — like a person
    # who reloads a page that failed to load. Deterministic errors (bad host, SSL,
    # blocked) are returned immediately; retrying them would just waste time.
    _TRANSIENT = ("err_timed_out", "err_connection_reset", "err_connection_closed",
                  "err_connection_aborted", "err_network_changed", "err_empty_response",
                  "timeout")
    last_err = None
    for attempt in range(2):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            last_err = None
            break
        except PWError as e:
            last_err = e
            if attempt == 0 and any(t in str(e).lower() for t in _TRANSIENT):
                await asyncio.sleep(1.0)
                continue
            break
    if last_err is not None:
        return {"status": "error", "msg": f"Navigation to {url} failed: {last_err}"}
    await _settle_nav(page)
    return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)


def _resolve_target(profile: str, session_id: str, tab_id: str,
                    element_id: int, expected: str):
    """Make expected_text the SOURCE OF TRUTH for which element to act on, not just a
    tripwire. Element ids are re-numbered by screen position on every observation, so
    an agent routinely pairs the right expected_text with a stale/wrong id. Behavior:
      - no expected_text, or no snapshot -> act on the id as given (nothing to check);
      - given id already matches expected_text -> keep it;
      - exactly ONE other element matches -> RETARGET to it (return a correction note);
      - MULTIPLE match -> ambiguous, return an error listing the candidate ids;
      - NONE match -> return an error (the wanted control isn't indexed on this page).
    Returns (resolved_id, note, error): at most one of note/error is set."""
    if not expected:
        return element_id, None, None
    snap = last_elements.get(_key(profile, session_id, tab_id), {})
    if not snap:
        return element_id, None, None  # no snapshot to check against; allow as-is
    exp = expected.lower()

    def _matches(info):
        return exp in f"{info.get('name', '')} {info.get('value', '')}".lower()

    info = snap.get(element_id)
    if info and _matches(info):
        return element_id, None, None  # the given id is already correct

    hits = [eid for eid, i in snap.items() if _matches(i)]
    if len(hits) == 1:
        rid = hits[0]
        ri = snap.get(rid, {})
        note = (f"[auto-corrected target] element [{element_id}] did not match expected_text "
                f"'{expected}'; retargeted to [{rid}] (role={ri.get('role')}, "
                f"name='{ri.get('name')}'). Element ids are re-numbered every observation — "
                f"prefer expected_text and re-read before acting.")
        return rid, note, None
    if len(hits) > 1:
        opts = ", ".join(f"[{h}] '{snap[h].get('name', '')}'" for h in hits[:8])
        err = (f"expected_text '{expected}' matches MULTIPLE elements: {opts}. "
               f"Re-issue targeting the specific [id] you want.")
        return element_id, None, err
    cur = snap.get(element_id, {})
    err = (f"expected_text '{expected}' not found in element [{element_id}] "
           f"(role={cur.get('role')}, name='{cur.get('name')}') NOR in any other indexed "
           f"element on this page. The control you want may not be captured — re-read the "
           f"page, scroll it into view, or target a different element.")
    return element_id, None, err


@app.post("/interact")
async def interact(req: InteractRequest):
    profile = _safe_profile(req.profile)
    ctx = await _ensure_browser(profile)
    page = await _get_page(profile, req.session_id, req.tab_id)
    a = (req.action or "").strip().lower()
    # 'select'/'choose' are common phrasings for choosing a dropdown option; accept
    # them as select_option, and let the agent pass the option in `text` (what a
    # 'type'-minded model reaches for) as well as the documented `value`.
    if a in ("select", "choose", "pick"):
        a = "select_option"
    if a == "select_option" and req.value is None and req.text is not None:
        req.value = req.text
    dialog_action = str(req.dialog_action or "auto").strip().lower()
    if dialog_action not in {"auto", "accept", "dismiss"}:
        raise HTTPException(
            status_code=400,
            detail="dialog_action must be auto, accept, or dismiss.",
        )
    request_dialog_policy = {
        "action": dialog_action,
        "text": req.dialog_text or "",
        "expires": time.monotonic() + 30.0,
    }
    dialog_policies[id(page)] = request_dialog_policy

    # ---- page-level actions that need no element ----
    try:
        if a in ("go_back", "back"):
            await page.go_back(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        if a in ("go_forward", "forward"):
            await page.go_forward(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        if a in ("reload", "refresh"):
            await page.reload(wait_until="domcontentloaded")
            await _settle_nav(page)
            return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        if a == "press_key":
            if not req.key:
                raise HTTPException(status_code=400, detail="press_key requires 'key'.")
            if req.element_id is not None:
                keyloc = await _locator(page, req.element_id, profile, req.session_id, req.tab_id)
                # Bring the pointer to the element first (mouse presence at the
                # target), then send the real keystroke to it. A hover, not a
                # click, so we never accidentally activate the element.
                try:
                    c = await _center(page, keyloc)
                    if c:
                        await keyloc.scroll_into_view_if_needed(timeout=5000)
                        await _human_move_to(page, c["x"], c["y"])
                except PWError:
                    pass
                await keyloc.press(req.key)
            else:
                await page.keyboard.press(req.key)
            await _settle_action(page)
            return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        if a == "scroll":
            await _do_scroll(page, req)
            await _settle_action(page)
            return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        if a == "wait_for":
            wait_note = await _do_wait(page, req)
            response = await _build_response(
                page, profile, req.session_id, req.tab_id, overlay=req.overlay
            )
            response.setdefault("events", []).insert(0, wait_note)
            return response
        if a == "read_text":
            # PDFs render in Chrome's viewer plugin, so their text is NOT in the
            # DOM. Fetch the bytes through the browser context (same cookies/proxy)
            # and save them so the agent can parse the file with a PDF tool.
            if page.url.lower().split("?")[0].split("#")[0].endswith(".pdf"):
                try:
                    resp = await page.context.request.get(page.url)
                    body = await resp.body()
                    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                    name = (page.url.split("/")[-1].split("?")[0].split("#")[0] or "document.pdf")
                    dest = os.path.join(DOWNLOAD_DIR, name)
                    with open(dest, "wb") as f:
                        f.write(body)
                    host = dest.replace("/profiles", "~/.aeon/browser_profiles", 1)
                    return {"text": (f"This page is a PDF; its text is not in the page DOM. Saved the "
                                     f"PDF ({len(body):,} bytes) to {host}. Parse it with a PDF tool via "
                                     f"run_command (e.g. `pdftotext {host} -` or a Python PDF library).")}
                except Exception as e:
                    return {"text": (f"This page is a PDF and could not be auto-saved ({e}). "
                                     f"Download it and parse with a PDF tool.")}
            # Clean main-content text (readability). Try the main document AND any
            # iframes (docs/readers/embeds often put the real article in a frame),
            # and return the richest result so embedded content isn't missed.
            best = ""
            for fr in page.frames[:MAX_FRAMES]:
                try:
                    t = await fr.evaluate(_READABILITY_JS)
                except Exception:
                    continue
                if t and len(t) > len(best):
                    best = t
            if not best:
                try:
                    best = await page.inner_text("body")
                except PWError:
                    best = ""
            return {"text": (best or "")[:14000]}
        if a in ("get_page_text", "get_text") and req.element_id is None:
            return {"text": (await page.inner_text("body"))[:14000]}

        # Screenshot-coordinate actions are the fallback for canvas, maps, remote
        # desktops, games, and controls with no useful DOM/accessibility node.
        # Coordinates are always validated against the CURRENT viewport, and the
        # DOM target beneath the point is surfaced as ground truth afterward.
        if a in {
            "click_at", "double_click_at", "right_click_at", "hover_at",
            "type_at", "drag_at", "press_and_hold_at",
        }:
            source_url = page.url
            point = await _coordinate_target(page, req.x, req.y)
            if a == "hover_at":
                await _human_move_to(page, point["x"], point["y"])
            elif a == "drag_at":
                end = await _coordinate_target(page, req.to_x, req.to_y)
                await _drag_points(page, point, end)
            elif a == "press_and_hold_at":
                await _human_move_to(page, point["x"], point["y"])
                await page.mouse.down()
                await asyncio.sleep(max(0, min(int(req.duration or 2000), 120000)) / 1000)
                await page.mouse.up()
            else:
                await _human_move_to(page, point["x"], point["y"])
                await _human_pause()
                button = "right" if a == "right_click_at" else "left"
                clicks = 2 if a == "double_click_at" else 1
                await page.mouse.click(
                    point["x"], point["y"], button=button, click_count=clicks,
                    delay=random.randint(40, 110),
                )
                if a == "type_at":
                    await _human_pause()
                    if req.clear_first:
                        await page.keyboard.press("Control+A")
                        await _human_pause(0.03, 0.09)
                        await page.keyboard.press("Delete")
                    await _human_type_text(
                        page, req.text or "", bool(req.then_enter)
                    )
            await _settle_action(page)
            resp = await _build_response(
                page, profile, req.session_id, req.tab_id, overlay=req.overlay
            )
            target = f"{point.get('tag') or 'unknown'}"
            if point.get("role"):
                target += f" role={point['role']}"
            if point.get("name"):
                target += f" name={point['name']!r}"
            resp.setdefault("events", []).insert(
                0, f"[coordinate target] ({point['x']:g}, {point['y']:g}) -> {target}",
            )
            if page.url == source_url:
                resp["action_focus"] = {
                    "label": target,
                    "source_url": source_url,
                    "rect": {"x": point["x"] - 2, "y": point["y"] - 2,
                             "w": 4, "h": 4},
                }
            return resp

        # ---- element-targeted actions ----
        if req.element_id is None:
            raise HTTPException(status_code=400, detail=f"action '{a}' requires element_id.")

        # Resolve the target from expected_text: ids are re-numbered by screen
        # position each observation, so an agent's id is often stale even when its
        # expected_text is right. For the actions where expected_text is honored,
        # RETARGET to the matching element (or fail with a helpful message), instead
        # of only rejecting a mismatch and leaving the agent to guess again.
        retarget_note = None
        if a in ("click", "double_click", "right_click", "type",
                 "check", "uncheck", "select_option", "clear", "upload_file"):
            resolved_id, retarget_note, rerr = _resolve_target(
                profile, req.session_id, req.tab_id, req.element_id, req.expected_text or "")
            if rerr:
                raise HTTPException(status_code=409, detail=rerr)
            req.element_id = resolved_id

        loc = await _locator(page, req.element_id, profile, req.session_id, req.tab_id)
        try:
            await loc.wait_for(state="attached", timeout=4000)
        except PWError:
            raise HTTPException(status_code=404,
                                detail=f"Element [{req.element_id}] is no longer on the page. Re-read it.")

        action_focus = None
        if a != "get_text":
            try:
                # Capture the resolved target before the action.  The post-action
                # extractor reassigns ids and therefore cannot safely recover this
                # rectangle after a re-render.
                await loc.scroll_into_view_if_needed(timeout=5000)
                box = await loc.bounding_box()
                if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
                    snap = last_elements.get(
                        _key(profile, req.session_id, req.tab_id), {}
                    ).get(req.element_id, {})
                    label = str(snap.get("name") or snap.get("value") or
                                f"element [{req.element_id}]")[:160]
                    action_focus = {
                        "label": label,
                        "source_url": page.url,
                        "rect": {"x": round(box["x"]), "y": round(box["y"]),
                                 "w": round(box["width"]), "h": round(box["height"])},
                    }
            except PWError:
                action_focus = None

        # Pre-click enabled guard: a coordinate click bypasses Playwright's enabled
        # check, so clicking a disabled control silently does nothing. Wait briefly
        # for an async validation to enable it; if it stays disabled, skip the dead
        # click and report WHY (surfaced as an event below) instead of misleading the
        # agent with an unchanged page.
        click_blocked_note = None
        if a in ("click", "double_click", "right_click"):
            click_blocked_note = await _wait_enabled_or_note(loc, req.element_id)

        if a == "click":
            if not click_blocked_note:
                await _human_click(page, loc)
        elif a == "double_click":
            if not click_blocked_note:
                await _human_click(page, loc, clicks=2)
        elif a == "right_click":
            if not click_blocked_note:
                await _human_click(page, loc, button="right")
        elif a == "hover":
            await loc.scroll_into_view_if_needed(timeout=5000)
            c = await _click_point(page, loc)
            if c:
                await _human_move_to(page, c["x"], c["y"])
            else:
                await loc.hover()
        elif a == "type":
            await _human_type(page, loc, req.text or "", bool(req.clear_first), bool(req.then_enter))
        elif a == "press_and_hold":
            await _press_and_hold(page, loc, req.duration or 2000)
        elif a == "select_option":
            if req.value is None:
                raise HTTPException(status_code=400,
                                    detail="select_option requires 'value' (the option value or visible label).")
            # Human approach: move to and click the <select> to focus/open it,
            # then pick. A native <select>'s option popup is OS-drawn and NOT in
            # the DOM, so the final pick must go through Playwright — but the
            # interaction still looks human (mouse moves to and clicks the field).
            try:
                await _human_click(page, loc)
            except PWError:
                pass
            # Match by the option's value attribute first; fall back to the VISIBLE
            # LABEL (what the tool doc promises and what the agent actually sees).
            try:
                await loc.select_option(req.value)
            except PWError:
                try:
                    await loc.select_option(label=req.value)
                except PWError:
                    # Not a native <select>. The click above already opened this
                    # custom dropdown, so its options are now in the DOM — guide the
                    # agent to click the option element by its [id].
                    raise HTTPException(
                        status_code=409,
                        detail=("select_option only works on a native <select>. This looks like a "
                                "custom dropdown — it has been opened; re-read the page and CLICK the "
                                "desired option by its [id]."))
        elif a == "check":
            await _human_set_checked(page, loc, True)
        elif a == "uncheck":
            await _human_set_checked(page, loc, False)
        elif a == "clear":
            await _human_clear(page, loc)
        elif a == "upload_file":
            if not req.file_path:
                raise HTTPException(status_code=400, detail="upload_file requires 'file_path'.")
            await _do_upload(page, loc, req.file_path)
        elif a == "drag":
            if req.to_element_id is None:
                raise HTTPException(status_code=400, detail="drag requires 'to_element_id'.")
            await _do_drag(page, req.element_id, req.to_element_id, profile, req.session_id, req.tab_id)
        elif a == "get_text":
            return {"text": await loc.inner_text()}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {a}")

        await _settle_action(page)
        resp = await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)
        # Surface an auto-retarget so the agent SEES its id was corrected (and learns
        # to lean on expected_text) instead of silently acting on a different element.
        if retarget_note:
            resp.setdefault("events", []).insert(0, retarget_note)
        # Surface a skipped disabled-target click so the agent gets the REASON the page
        # did not change, rather than inferring "rejected/taken" and looping.
        if click_blocked_note:
            resp.setdefault("events", []).insert(0, click_blocked_note)
        if action_focus and page.url == action_focus["source_url"]:
            resp["action_focus"] = action_focus
        return resp

    except HTTPException:
        raise
    except PWError as e:
        raise HTTPException(status_code=500, detail=f"Browser action '{a}' failed: {e}")
    finally:
        # Do not let an explicit response leak into a later unrelated dialog if
        # this action never opened one. Identity-check protects a newer request.
        if dialog_policies.get(id(page)) is request_dialog_policy:
            dialog_policies.pop(id(page), None)


# Scrolls the nearest scrollable ancestor of an element on the requested axis
# (falls back to the window). Handles BOTH vertical (top/bottom) and horizontal
# (left/right) — an element inside an iframe scrolls in its own frame.
_CONTAINER_SCROLL_JS = r"""
({id, amount, axis, sign, toEnd}) => {
  const el = document.querySelector('[data-aeon-id="' + id + '"]');
  if (!el) return;
  const vert = axis === 'y';
  let p = el.parentElement;
  while (p && p !== document.body) {
    const s = getComputedStyle(p);
    const ov = vert ? s.overflowY : s.overflowX;
    const can = vert ? (p.scrollHeight > p.clientHeight + 4) : (p.scrollWidth > p.clientWidth + 4);
    if ((ov === 'auto' || ov === 'scroll') && can) {
      if (vert) { p.scrollTop = toEnd ? (sign > 0 ? p.scrollHeight : 0) : p.scrollTop + sign * amount; }
      else      { p.scrollLeft = toEnd ? (sign > 0 ? p.scrollWidth : 0)  : p.scrollLeft + sign * amount; }
      return;
    }
    p = p.parentElement;
  }
  if (vert) window.scrollBy(0, sign * amount); else window.scrollBy(sign * amount, 0);
}
"""

# direction -> (axis, sign, to-end?). Vertical: down/up/bottom/top.
# Horizontal: right/left/rightmost/leftmost.
_SCROLL_DIRS = {
    "down": ("y", 1, False), "up": ("y", -1, False),
    "bottom": ("y", 1, True), "top": ("y", -1, True),
    "right": ("x", 1, False), "left": ("x", -1, False),
    "rightmost": ("x", 1, True), "leftmost": ("x", -1, True),
}


async def _do_scroll(page: Page, req: InteractRequest):
    profile = _safe_profile(req.profile)
    amount = abs(req.amount) if req.amount is not None else 600
    direction = (req.direction or "down").lower()

    if req.element_id is not None and direction in ("to_element", "into_view"):
        await (await _locator(page, req.element_id, profile, req.session_id, req.tab_id)).scroll_into_view_if_needed(timeout=5000)
        return

    axis, sign, to_end = _SCROLL_DIRS.get(direction, ("y", 1, False))

    if req.element_id is not None:
        # Scroll the scroll-container holding the element — IN ITS OWN FRAME, so
        # an inbox/list/chat/table pane (incl. inside an iframe) scrolls correctly.
        owner = _frame_for(profile, req.session_id, req.tab_id, req.element_id, page)
        await owner.evaluate(_CONTAINER_SCROLL_JS,
                             {"id": req.element_id, "amount": amount,
                              "axis": axis, "sign": sign, "toEnd": to_end})
        return

    # Page-level scroll.
    if to_end:
        if axis == "y":
            target = "document.body.scrollHeight" if sign > 0 else "0"
            await page.evaluate(f"() => window.scrollTo(window.scrollX, {target})")
        else:
            target = "document.body.scrollWidth" if sign > 0 else "0"
            await page.evaluate(f"() => window.scrollTo({target}, window.scrollY)")
        return
    # Incremental: roll the wheel in human-sized notches (vertical OR horizontal)
    # rather than one teleporting jump; also lets lazy content stream in naturally.
    for tick in scroll_ticks(sign * amount):
        if axis == "y":
            await page.mouse.wheel(0, tick)
        else:
            await page.mouse.wheel(tick, 0)
        await asyncio.sleep(random.uniform(0.03, 0.09))


async def _do_upload(page: Page, loc, file_path: str):
    """Upload a file. If the element is a real <input type=file>, set it directly;
    otherwise it's a button that opens the OS file chooser — intercept that."""
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail=f"upload_file: '{file_path}' does not exist in the browser environment. "
                   f"The file must be reachable from inside the browser container.")
    try:
        tag = (await loc.evaluate("el => el.tagName.toLowerCase() + '|' + (el.type||'')")).split("|")
    except PWError:
        tag = ["", ""]
    if tag[0] == "input" and tag[1] == "file":
        await loc.set_input_files(file_path)
        return
    # A button that opens the OS chooser. Bound the wait so a mis-targeted element
    # (one that never opens a chooser) fails fast with a clear message instead of
    # hanging on Playwright's default 30s timeout.
    try:
        async with page.expect_file_chooser(timeout=8000) as fc_info:
            await _human_click(page, loc)   # human motion opens the picker
        chooser = await fc_info.value
    except Exception:
        raise HTTPException(
            status_code=409,
            detail="upload_file: clicking that element did not open a file chooser within 8s. "
                   "Target the actual file-input or the button that opens the upload dialog.")
    await chooser.set_files(file_path)


async def _do_drag(page: Page, src_id: int, dst_id: int, profile: str = None,
                   session_id: str = None, tab_id: str = None):
    src = await _locator(page, src_id, profile, session_id, tab_id)
    dst = await _locator(page, dst_id, profile, session_id, tab_id)
    # Bring the source into view first so its coordinates are on-screen (a drag
    # needs both endpoints visible; the target is expected to be nearby).
    try:
        await src.scroll_into_view_if_needed(timeout=5000)
    except PWError:
        pass
    sc = await _center(page, src)
    dc = await _center(page, dst)
    if not sc or not dc:
        raise HTTPException(status_code=500,
                            detail="Could not locate drag source/target geometry (both must be on-screen; "
                                   "scroll them into view first).")
    # Humans press, pause, THEN drag; _drag_points provides the same bounded,
    # time-sampled path for element and coordinate drags.
    await _drag_points(page, sc, dc)


async def _do_wait(page: Page, req: InteractRequest) -> str:
    started = time.monotonic()
    if req.text:
        try:
            await page.get_by_text(req.text, exact=False).first.wait_for(timeout=(req.duration or 10000))
        except PWError as exc:
            elapsed = time.monotonic() - started
            return (
                f"[wait:TIMEOUT after {elapsed:.1f}s] text {req.text!r} did not appear; "
                f"condition was NOT satisfied ({str(exc).splitlines()[0][:120]})."
            )
        elapsed = time.monotonic() - started
        return f"[wait:satisfied after {elapsed:.1f}s] text {req.text!r} appeared."
    else:
        delay_ms = max(0, min(int(req.duration or 2000), 120000))
        await asyncio.sleep(delay_ms / 1000)
        return f"[wait:completed] paused for {delay_ms / 1000:.1f}s."


@app.post("/observe")
async def observe(req: TabRequest):
    profile = _safe_profile(req.profile)
    await _ensure_browser(profile)
    page = await _get_page(profile, req.session_id, req.tab_id)
    # A pure re-look (browser_read) is the safe moment for an idle 'reading'
    # cursor wander — no following action to disturb (unlike before an interaction,
    # where a drift could dismiss a just-opened hover menu). Real actions already
    # provide abundant curved pointer motion on their own.
    if random.random() < 0.5:
        try:
            await _human_idle_drift(page)
        except Exception:
            pass
    return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)


@app.post("/find")
async def find_on_page(req: FindRequest):
    """Return a filtered, actionable observation for semantic page matches."""
    profile = _safe_profile(req.profile)
    await _ensure_browser(profile)
    page = await _get_page(profile, req.session_id, req.tab_id)
    query = (req.text or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="find requires non-empty text")
    return await _build_response(
        page, profile, req.session_id, req.tab_id, overlay=req.overlay,
        search_text=query[:300], role_filter=(req.role or "")[:80],
    )


@app.post("/extract")
async def extract_page(req: ExtractRequest):
    """Return grounded text or structured links/forms/tables across all frames."""
    profile = _safe_profile(req.profile)
    await _ensure_browser(profile)
    page = await _get_page(profile, req.session_id, req.tab_id)
    mode = (req.mode or "").strip().lower()
    if mode not in {"text", "links", "forms", "tables"}:
        raise HTTPException(status_code=400, detail="extract mode must be text, links, forms, or tables")
    limit = max(1, min(int(req.max_items or 200), 500))
    if mode == "text":
        best = ""
        for frame in page.frames[:MAX_FRAMES]:
            try:
                value = await frame.evaluate(_READABILITY_JS)
            except Exception:
                continue
            if value and len(value) > len(best):
                best = value
        if not best:
            try:
                best = await page.inner_text("body")
            except PWError:
                best = ""
        return {"text": (best or "")[:50000]}

    frames = []
    remaining = limit
    for frame in page.frames[:MAX_FRAMES]:
        if remaining <= 0:
            break
        try:
            items = await frame.evaluate(
                _STRUCTURED_EXTRACT_JS,
                {"mode": mode, "maxItems": remaining},
            )
        except Exception:
            continue
        if items:
            frames.append({"frame_url": frame.url, "items": items})
            # Tables are grouped objects; count their rows toward the requested
            # bound. Other modes have one array item per extracted entity.
            if mode == "tables":
                used = sum(len(t.get("rows") or []) for t in items)
            else:
                used = len(items)
            remaining -= max(1, used)
    result = {
        "mode": mode,
        "page_url": page.url,
        "truncated": remaining <= 0,
        "frames": frames,
    }
    return {"text": json.dumps(result, ensure_ascii=False, indent=2)[:50000]}


# --- Media capture (save an image/video the agent sees on the page) ----------
# The agent perceives media in the SCREENSHOT, but <img>/<video> are not in the
# interactive element index (only clickable/semantic roles are). So capture has
# its own enumeration: stamp every media node with data-aeon-media=<id>, hand the
# agent the list, and let it pick one by id. Then download the ORIGINAL bytes when
# a real URL exists (best), else fall back to re-rendering pixels — for images an
# element screenshot, for video yt-dlp then a screen recording.
_ENUM_MEDIA_JS = r"""
() => {
  const out = [];
  let id = 0;
  const vw = innerWidth, vh = innerHeight;
  const add = (el, tag, src, natW, natH) => {
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;          // skip trackers/spacers/icons
    el.setAttribute('data-aeon-media', String(id));
    out.push({
      id, tag, src: src || '',
      w: Math.round(natW || r.width), h: Math.round(natH || r.height),
      dw: Math.round(r.width), dh: Math.round(r.height),
      x: Math.round(r.left), y: Math.round(r.top),
      area: Math.round(r.width * r.height),
      alt: (el.getAttribute('alt') || el.getAttribute('aria-label') ||
            el.getAttribute('title') || '').replace(/\s+/g,' ').trim().slice(0,120),
      inView: r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw,
    });
    id++;
  };
  document.querySelectorAll('img').forEach(el => add(el, 'img', el.currentSrc || el.src, el.naturalWidth, el.naturalHeight));
  document.querySelectorAll('video').forEach(el => add(el, 'video', el.currentSrc || el.src || '', el.videoWidth, el.videoHeight));
  document.querySelectorAll('canvas').forEach(el => add(el, 'canvas', '', el.width, el.height));
  // CSS background images (hero banners, some ads), bounded so a huge DOM stays cheap.
  const all = document.querySelectorAll('body *');
  for (let i = 0; i < all.length && out.length < 60; i++) {
    const el = all[i];
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg.indexOf('url(') === 0) {
      const m = bg.match(/url\(["']?(.*?)["']?\)/);
      if (m && m[1] && m[1].indexOf('data:image/svg') !== 0) add(el, 'bg', m[1]);
    }
  }
  return out;
}
"""

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".m4v", ".mkv")


def _unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}{ext}"


def _ext_from(src: str, content_type: str, default: str) -> str:
    path = src.split("?")[0].split("#")[0]
    e = os.path.splitext(path)[1].lower()
    if e and len(e) <= 5:
        return e
    ct = (content_type or "").split(";")[0].strip().lower()
    ct_map = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
              "image/webp": ".webp", "image/svg+xml": ".svg", "video/mp4": ".mp4",
              "video/webm": ".webm"}
    return ct_map.get(ct, default)


async def _fetch_bytes(page: Page, src: str):
    """Download a media URL THROUGH the page's own context (its cookies/proxy), so
    auth-gated assets work. Returns (bytes, content_type) or (None, None)."""
    try:
        resp = await page.context.request.get(src, timeout=45000)
        if not resp.ok:
            return None, None
        body = await resp.body()
        ct = ""
        try:
            ct = (await resp.header_value("content-type")) or ""
        except Exception:
            pass
        return (body or None), ct
    except Exception:
        return None, None


async def _capture_image(page: Page, entry: dict, out_dir: str) -> Dict[str, Any]:
    src = entry.get("src") or ""
    prefix = "capture_img"
    # 1. Original bytes when a real URL exists — the actual file, full quality.
    if src.startswith("http"):
        body, ct = await _fetch_bytes(page, src)
        if body:
            dest = os.path.join(out_dir, _unique_name(prefix, _ext_from(src, ct, ".jpg")))
            with open(dest, "wb") as f:
                f.write(body)
            return {"filename": os.path.basename(dest), "method": "downloaded original image bytes"}
    # 2. Fallback: re-render the exact node's pixels (canvas / blob: / data: / CSS bg,
    #    or a fetch that failed). Always works because it screenshots what's on screen.
    dest = os.path.join(out_dir, _unique_name(prefix, ".png"))
    loc = page.locator(f'[data-aeon-media="{entry["id"]}"]').first
    await loc.scroll_into_view_if_needed(timeout=5000)
    await loc.screenshot(path=dest)
    return {"filename": os.path.basename(dest), "method": "element screenshot (re-rendered pixels, not the original file)"}


async def _run(cmd: list, timeout: int) -> tuple:
    """Run a subprocess without blocking the event loop; return (rc, stderr_tail)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, (err or b"").decode("utf-8", "ignore")[-400:]
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 124, f"timed out after {timeout}s"


async def _capture_video(page: Page, entry: dict, out_dir: str, duration_s: int) -> Dict[str, Any]:
    src = entry.get("src") or ""
    prefix = "capture_vid"
    # 1. Direct video file URL -> download the original bytes.
    if src.startswith("http") and any(src.split("?")[0].lower().endswith(e) for e in _VIDEO_EXTS):
        body, ct = await _fetch_bytes(page, src)
        if body:
            dest = os.path.join(out_dir, _unique_name(prefix, _ext_from(src, ct, ".mp4")))
            with open(dest, "wb") as f:
                f.write(body)
            return {"filename": os.path.basename(dest), "method": "downloaded original video file"}
    # 2. yt-dlp on the PAGE url — handles HLS/DASH/blob players, YouTube, most embeds.
    stem = _unique_name(prefix, "")
    rc, err = await _run(
        ["yt-dlp", "--no-playlist", "--no-warnings", "-f", "mp4/best",
         "-o", os.path.join(out_dir, stem + ".%(ext)s"), page.url],
        timeout=max(60, duration_s * 6))
    hits = [f for f in os.listdir(out_dir) if f.startswith(stem)]
    if rc == 0 and hits:
        return {"filename": hits[0], "method": "downloaded via yt-dlp (page media stream)"}
    # 3. Last resort: screen-record the live Xvfb display while the video plays.
    dest = os.path.join(out_dir, _unique_name(prefix, ".mp4"))
    display = os.environ.get("DISPLAY", ":99")
    rc, err = await _run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0",
         "-video_size", f"{SCREEN_W}x{SCREEN_H}", "-framerate", "24", "-i", display,
         "-t", str(max(1, min(int(duration_s), 60))), "-pix_fmt", "yuv420p", dest],
        timeout=max(30, int(duration_s) + 20))
    if rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return {"filename": os.path.basename(dest),
                "method": f"screen recording of the page for {duration_s}s (not the original file)"}
    raise HTTPException(status_code=502,
                        detail=f"Could not download the video and the screen-recording fallback failed ({err}).")


@app.post("/capture_media")
async def capture_media(req: CaptureRequest):
    profile = _safe_profile(req.profile)
    await _ensure_browser(profile)
    page = await _get_page(profile, req.session_id, req.tab_id)
    try:
        media = await page.evaluate(_ENUM_MEDIA_JS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not enumerate page media: {e}")
    media.sort(key=lambda m: m.get("area", 0), reverse=True)  # biggest first (most likely intended)

    # No id chosen -> return the catalog for the agent to pick from.
    if req.media_id is None:
        return {"status": "ok", "mode": "list", "url": page.url, "media": media[:40]}

    entry = next((m for m in media if m.get("id") == req.media_id), None)
    if entry is None:
        raise HTTPException(status_code=404,
                            detail=f"No media with id {req.media_id} on this page. Call capture_media without media_id to re-list.")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if entry["tag"] == "video":
        result = await _capture_video(page, entry, DOWNLOAD_DIR, int(req.duration_s or 8))
    else:
        result = await _capture_image(page, entry, DOWNLOAD_DIR)
    return {"status": "ok", "mode": "capture", "media_id": req.media_id,
            "tag": entry["tag"], **result}


@app.post("/switch_tab")
async def switch_tab(req: TabRequest):
    profile = _safe_profile(req.profile)
    await _ensure_browser(profile)
    page = None
    if _key(profile, req.session_id, req.tab_id) in pages:
        page = pages[_key(profile, req.session_id, req.tab_id)]
    elif _key(profile, "_popups", req.tab_id) in pages:
        # Adopt a captured popup into the caller's session so subsequent
        # browser_interact/browser_read on this tab_id resolve to it.
        page = pages.pop(_key(profile, "_popups", req.tab_id))
        pages[_key(profile, req.session_id, req.tab_id)] = page
    if page is None:
        raise HTTPException(status_code=404, detail=f"Tab '{req.tab_id}' not found.")
    if page.is_closed():
        raise HTTPException(status_code=404,
                            detail=f"Tab '{req.tab_id}' has closed. Switch to another open tab or navigate to re-open it.")
    try:
        await page.bring_to_front()
    except PWError:
        pass  # page may have closed; _build_response will surface the real state
    # A just-opened popup often sits at about:blank for a moment BEFORE it starts
    # navigating to its real URL. Wait briefly for it to leave about:blank so we
    # report the loaded page, not a blank one. A genuinely blank popup just falls
    # through after the short poll; then _settle_nav waits for the load (bounded).
    for _ in range(15):  # ~3s max
        try:
            if page.url not in ("about:blank", ""):
                break
        except PWError:
            break
        await asyncio.sleep(0.2)
    await _settle_nav(page)
    return await _build_response(page, profile, req.session_id, req.tab_id, overlay=req.overlay)


@app.post("/close_tab")
async def close_tab(req: TabRequest):
    profile = _safe_profile(req.profile)
    k = _key(profile, req.session_id, req.tab_id)
    pk = _key(profile, "_popups", req.tab_id)
    target = k if k in pages else (pk if pk in pages else None)
    if target:
        pg = pages[target]
        try:
            await pg.close()
        except PWError:
            pass
        # Drop ALL per-page state so a long session that opens/closes many tabs
        # doesn't slowly leak snapshots, frame maps, and event lists.
        del pages[target]
        last_elements.pop(target, None)
        frame_maps.pop(target, None)
        page_events.pop(id(pg), None)
        _announced_popups.discard(target)
    return {"status": "success", "remaining_tabs": len(_tabs_for(profile, req.session_id))}


@app.post("/release_profile")
async def release_profile(req: ProfileRequest):
    """Close and forget an isolated profile's context, freeing its Chrome — called
    by a sub-agent when it finishes so its browser doesn't linger. The shared
    'default' profile is never torn down here (the principal keeps using it)."""
    profile = _safe_profile(req.profile)
    if profile == DEFAULT_PROFILE:
        return {"status": "kept", "reason": "default profile is shared/persistent"}
    ctx = contexts.pop(profile, None)
    _purge_profile_state(profile)
    _launch_locks.pop(profile, None)
    if ctx is not None:
        try:
            await ctx.close()
        except Exception:
            pass
    # Delete the on-disk profile dir so ephemeral sub-agent profiles don't pile
    # up. Safe: the context is closed, and only NON-default profiles reach here.
    try:
        import shutil
        shutil.rmtree(_profile_dir(profile), ignore_errors=True)
    except Exception:
        pass
    return {"status": "released", "profile": profile}


@app.on_event("shutdown")
async def on_shutdown():
    global _playwright
    try:
        for ctx in list(contexts.values()):
            try:
                await ctx.close()
            except Exception:
                pass
        contexts.clear()
        if _playwright:
            await _playwright.stop()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8030")))
