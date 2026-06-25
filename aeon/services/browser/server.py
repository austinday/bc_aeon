import asyncio
import json
import logging
import os
import random
import time
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("browser_server")

app = FastAPI()

# Global state
browser_instance: Optional[Browser] = None
contexts: Dict[str, BrowserContext] = {}
pages: Dict[str, Page] = {}

class SessionState:
    def __init__(self, page: Page):
        self.page = page
        self.som_elements: List[Dict[str, Any]] = []

session_states: Dict[str, SessionState] = {}

class InteractRequest(BaseModel):
    session_id: str
    action: str
    element_id: Optional[int] = None
    selector: Optional[str] = None
    duration: Optional[int] = 2000
    text: Optional[str] = None
    expected_text: Optional[str] = None

async def get_page(session_id: str) -> Page:
    if session_id not in pages:
        raise HTTPException(status_code=404, detail="Session not found")
    return pages[session_id]

async def human_jitter_hold(page: Page, selector: str, duration_ms: int):
    """
    Performs a press-and-hold with human-like jitter.
    """
    element = await page.wait_for_selector(selector)
    box = await element.bounding_box()
    if not box:
        raise Exception("Could not get bounding box")

    center_x = box['x'] + box['width'] / 2
    center_y = box['y'] + box['height'] / 2

    # Move to element
    await page.mouse.move(center_x, center_y)
    
    # Press down
    await page.mouse.down()
    
    start_time = time.time()
    while (time.time() - start_time) * 1000 < duration_ms:
        # Small random movements (jitter)
        jx = center_x + random.uniform(-2, 2)
        jy = center_y + random.uniform(-2, 2)
        await page.mouse.move(jx, jy)
        await asyncio.sleep(0.05)
    
    await page.mouse.up()

@app.post("/interact")
async def interact(req: InteractRequest):
    logger.info(f"Interact Request: {req.dict()}")
    
    # FIX: Allow selector-based or page-wide actions without element_id
    if req.action in ["get_page_text", "get_text_by_selector"]:
        if req.action == "get_page_text":
            page = await get_page(req.session_id)
            return {"text": await page.content()}
        if req.action == "get_text_by_selector":
            if not req.selector:
                raise HTTPException(status_code=400, detail="Selector required for get_text_by_selector")
            page = await get_page(req.session_id)
            return {"text": await page.inner_text(req.selector)}

    # Dispatcher for element-based actions
    if req.element_id is None:
        # If it's not a page-wide action, it MUST have an element_id
        raise HTTPException(status_code=400, detail="Invalid action or missing element_id")

    try:
        state = session_states.get(req.session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Session state not found")
        
        # Find element in SOM list
        element_data = next((e for e in state.som_elements if e['id'] == req.element_id), None)
        if not element_data:
            raise HTTPException(status_code=404, detail=f"Element ID {req.element_id} not found in DOM")
        
        selector = element_data['selector']
        page = state.page

        if req.action == "click":
            await page.click(selector)
        elif req.action == "type":
            await page.fill(selector, req.text)
        elif req.action == "press_and_hold":
            await human_jitter_hold(page, selector, req.duration or 2000)
        elif req.action == "get_text":
            return {"text": await page.inner_text(selector)}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {req.action}")
            
        return {"status": "success"}
    except Exception as e:
        logger.exception("Error during interaction")
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS (Bypass Dispatcher and SOM) ---

@app.post("/debug_press_and_hold")
async def debug_press_and_hold(req: InteractRequest):
    """Bypasses SOM and dispatcher. Uses 'selector' directly."""
    if not req.selector:
        raise HTTPException(status_code=400, detail="Selector required")
    page = await get_page(req.session_id)
    await human_jitter_hold(page, req.selector, req.duration or 2000)
    return {"status": "success"}

@app.post("/debug_get_text")
async def debug_get_text(req: InteractRequest):
    """Bypasses SOM and dispatcher. Uses 'selector' directly."""
    if not req.selector:
        raise HTTPException(status_code=400, detail="Selector required")
    page = await get_page(req.session_id)
    text = await page.inner_text(req.selector)
    return {"text": text}

@app.post("/debug_get_page_text")
async def debug_get_page_text(req: InteractRequest):
    """Bypasses SOM and dispatcher. Returns full page text."""
    page = await get_page(req.session_id)
    return {"text": await page.content()}

@app.get("/ping")
async def ping():
    return {"status": "pong", "version": "jitter_v1"}

# --- Screenshot & Page Data Helpers ---

async def capture_screenshot(page: Page) -> Tuple[str, str]:
    """Capture full page screenshot and return base64 encoded clean and overlay images."""
    # Capture clean screenshot
    clean_bytes = await page.screenshot(full_page=True, type='jpeg', quality=85)
    clean_b64 = base64.b64encode(clean_bytes).decode('utf-8')
    
    # For overlay, we'll use the same screenshot for now (SOM overlay will be added later)
    overlay_b64 = clean_b64
    
    return clean_b64, overlay_b64

async def extract_page_data(page: Page) -> Dict[str, Any]:
    """Extract page title, URL, visible text, and interactive elements."""
    title = await page.title()
    url = page.url
    
    # Get visible text (simplified markdown)
    markdown = await page.inner_text('body')
    # Truncate very long text
    if len(markdown) > 10000:
        markdown = markdown[:10000] + "\n\n[... content truncated ...]"
    
    # Build interactive elements list with robust selectors
    elements = await page.query_selector_all(
        "button, input, a, [role='button'], [role='link'], select, textarea, "
        "[tabindex]:not([tabindex='-1']), [onclick], [contenteditable='true'], "
        "label, summary, details, [type='checkbox'], [type='radio'], [type='submit']"
    )
    
    som_elements = []
    for i, el in enumerate(elements):
        try:
            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            text = await el.inner_text()
            text = text.strip()[:100] if text else ""
            
            # Build a robust CSS selector using attributes
            selector = await el.evaluate("""el => {
                // Try ID first
                if (el.id) return '#' + CSS.escape(el.id);
                // Try unique class combination
                if (el.className && typeof el.className === 'string' && el.className.trim()) {
                    const classes = el.className.trim().split(/\\s+/).slice(0, 3).map(c => CSS.escape(c)).join('.');
                    return tag + '.' + classes;
                }
                // Fallback: use nth-of-type path
                let path = tag;
                let parent = el.parentElement;
                while (parent && parent !== document.body) {
                    const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
                    if (siblings.length > 1) {
                        const idx = siblings.indexOf(el) + 1;
                        path = tag + ':nth-of-type(' + idx + ')' + ' > ' + path;
                    } else {
                        path = tag + ' > ' + path;
                    }
                    el = parent;
                    parent = parent.parentElement;
                }
                return path;
            }""")
            
            som_elements.append({
                "id": i + 1,
                "tag": tag,
                "text": text,
                "selector": selector
            })
        except Exception:
            # Skip elements that can't be evaluated
            pass
    
    return {
        "title": title,
        "url": url,
        "markdown": markdown,
        "elements": som_elements
    }

async def build_response(page: Page, session_id: str, tab_id: str) -> Dict[str, Any]:
    """Build the full response with screenshots, page data, and open tabs."""
    clean_b64, overlay_b64 = await capture_screenshot(page)
    page_data = await extract_page_data(page)
    
    # Get list of open tabs for this session
    open_tabs = list(pages.keys())
    
    return {
        "status": "success",
        "session_id": session_id,
        "tab_id": tab_id,
        "clean_b64": clean_b64,
        "overlay_b64": overlay_b64,
        "title": page_data["title"],
        "url": page_data["url"],
        "markdown": page_data["markdown"],
        "elements": page_data["elements"],
        "open_tabs": open_tabs
    }

# --- Standard Browser Management ---

async def start_browser():
    global browser_instance
    playwright = await async_playwright().start()
    browser_instance = await playwright.chromium.launch(headless=True)
    return browser_instance

@app.post("/navigate")
async def navigate(req: Dict[str, Any]):
    session_id = req.get("session_id", "default")
    url = req.get("url")
    
    if session_id not in contexts:
        ctx = await browser_instance.new_context()
        contexts[session_id] = ctx
        page = await ctx.new_page()
        pages[session_id] = page
        session_states[session_id] = SessionState(page)
    
    page = pages[session_id]
    await page.goto(url)
    
    # Wait for page to fully load
    await page.wait_for_load_state("networkidle")
    
    return await build_response(page, session_id, req.get("tab_id", "default"))

@app.post("/switch_tab")
async def switch_tab(req: Dict[str, Any]):
    session_id = req.get("session_id", "default")
    tab_id = req.get("tab_id", "default")
    
    if session_id not in pages:
        raise HTTPException(status_code=404, detail="Session not found")
    
    page = pages[session_id]
    return await build_response(page, session_id, tab_id)

@app.post("/close_tab")
async def close_tab(req: Dict[str, Any]):
    session_id = req.get("session_id", "default")
    tab_id = req.get("tab_id", "default")
    
    if session_id in pages:
        page = pages[session_id]
        await page.close()
        del pages[session_id]
        if session_id in contexts:
            await contexts[session_id].close()
            del contexts[session_id]
        if session_id in session_states:
            del session_states[session_id]
    
    remaining = len(pages)
    return {"status": "success", "remaining_tabs": remaining}

@app.on_event("startup")
async def on_startup():
    await start_browser()

@app.on_event("shutdown")
async def on_shutdown():
    if browser_instance:
        await browser_instance.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)