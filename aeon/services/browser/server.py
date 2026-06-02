import asyncio
import json
import logging
import os
import random
import time
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
    # DEBUG: Log all incoming requests to see why they are failing
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

# --- DEBUG ENDPOINTS (Bypass Dispatcher and SOM) ---

@app.post("/debug_press_and_hold")
async def debug_press_and_hold(req: InteractRequest):
    """Bypasses SOM and dispatcher. Uses 'selector' directly."""
    logger.info(f"DEBUG Press-and-Hold: {req.selector} for {req.duration}ms")
    if not req.selector:
        raise HTTPException(status_code=400, detail="Selector required")
    page = await get_page(req.session_id)
    await human_jitter_hold(page, req.selector, req.duration or 2000)
    return {"status": "success"}

@app.post("/debug_get_text")
async def debug_get_text(req: InteractRequest):
    """Bypasses SOM and dispatcher. Uses 'selector' directly."""
    logger.info(f"DEBUG Get Text: {req.selector}")
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
    
    # Update SOM
    await update_som(session_id)
    
    return {"status": "success", "session_id": session_id}

async def update_som(session_id: str):
    page = pages[session_id]
    state = session_states[session_id]
    
    # Simple SOM implementation: find all buttons, inputs, etc.
    elements = await page.query_selector_all("button, input, a, [role='button'], div[onclick]")
    state.som_elements = []
    for i, el in enumerate(elements):
        # This is a simplification. In reality, we'd use a more robust selector.
        # For this debug version, we'll just use a unique identifier if possible.
        selector = f"div:nth-of-type({i+1})" # Very brittle, but for demo
        # Better: use a custom attribute or just the index
        state.som_elements.append({"id": i + 1, "selector": selector})

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