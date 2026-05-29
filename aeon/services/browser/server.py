import asyncio
import base64
import sys
import random
import math
import numpy as np

print("Loading imports...", flush=True)
from fastapi import FastAPI
from pydantic import BaseModel
from camoufox.async_api import AsyncCamoufox

print("Imports loaded. Initializing FastAPI...", flush=True)
app = FastAPI()
browser_instance = None
contexts = {}  # session_id -> context
tabs = {}       # session_id_tab_id -> page
session_popup_counts = {} # session_id -> int
popup_lock = asyncio.Lock()
@app.on_event("startup")
async def startup():
    global browser_instance
    print("Starting Camoufox browser instance...", flush=True)
    try:
        # Launch Camoufox: Upgrade to 1920x1080 to prevent tablet-layout wrapping
        browser_instance = await AsyncCamoufox(
            headless=False,
            geoip=False,
            args=[
                '--width=1920', 
                '--height=1080',
                '--window-position=0,0'
            ]
        ).__aenter__()
        print("Camoufox started successfully.", flush=True)
    except Exception as e:
        print(f"CRITICAL: Failed to start Camoufox: {e}", file=sys.stderr, flush=True)

@app.on_event("shutdown")
async def shutdown():
    global browser_instance
    if browser_instance:
        await browser_instance.__aexit__(None, None, None)

async def handle_popup(page):
    """Callback to track new pages/popups created by the browser."""
    print(f"DEBUG: [POPUP-EVENT] Page event fired for page object {id(page)}", flush=True)
    
    # If the page is already tracked, ignore it
    if page in tabs.values():
        # Find which tab it is
        existing_key = next((k for k, p in tabs.items() if p == page), "unknown")
        print(f"DEBUG: [POPUP-IGNORE] Page {id(page)} already tracked as {existing_key}", flush=True)
        return

    # Find session_id from page.context
    session_id = None
    for sid, ctx in contexts.items():
        if ctx == page.context:
            session_id = sid
            break
    
    if not session_id:
        print(f"DEBUG: [POPUP-ERROR] Popup detected but no matching session_id found for context {page.context}", flush=True)
        return

    async with popup_lock:
        # Use a persistent counter per session to avoid ID collisions and race conditions
        count = session_popup_counts.get(session_id, 0) + 1
        session_popup_counts[session_id] = count
        
        tab_id = f"popup_{count}"
        key = f"{session_id}_{tab_id}"
        tabs[key] = page
        print(f"DEBUG: [POPUP-TRACKED] New popup tracked: {key} (Session: {session_id}, Count: {count}, PageObj: {id(page)})", flush=True)

async def get_or_create_session(session_id: str, tab_id: str):
    global browser_instance, contexts, tabs
    
    if session_id not in contexts:
        # device_scale_factor=2 simulates a high-DPI (Retina) display, doubling screenshot resolution
        ctx = await browser_instance.new_context(device_scale_factor=2)
        contexts[session_id] = ctx
        
        # Capture console logs for debugging
        ctx.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}", flush=True))
        
        # Listen for new pages (popups) and track them automatically
        ctx.on("page", lambda page: asyncio.create_task(handle_popup(page)))
    
    key = f"{session_id}_{tab_id}"
    if key not in tabs:
        page = await contexts[session_id].new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})
        
        # ROBUST LAYOUT ENFORCEMENT: Force CSS reset to prevent massive left-margins 
        # or content shifting that pushes elements off-screen in headless mode.
        await page.add_init_script("""
            window.addEventListener('DOMContentLoaded', () => {
                const style = document.createElement('style');
                style.textContent = 'html, body { max-width: 100vw !important; overflow-x: hidden !important; margin: 0 !important; padding: 0 !important; }';
                document.head.appendChild(style);
            });
        """)
        
        tabs[key] = page
        
        # RACE CONDITION FIX: If the 'page' event fired and tracked this main page as a popup 
        # before we could add it to tabs, remove that duplicate entry.
        popup_keys = [k for k, p in tabs.items() if p == page and 'popup_' in k]
        for pk in popup_keys:
            del tabs[pk]
            
    return tabs[key]

class GotoRequest(BaseModel):
    url: str
    session_id: str
    tab_id: str = "default"

class SwitchTabRequest(BaseModel):
    session_id: str
    tab_id: str

@app.post("/switch_tab")
async def switch_tab(req: SwitchTabRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        # Removed bring_to_front to avoid driver crashes in some environments
        return await extract_page_state(page, req.session_id)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.post("/navigate")
async def navigate(req: GotoRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        # Removed bring_to_front to avoid driver crashes
        
        # Increased timeout and changed to networkidle to better handle heavy SPAs and bot-protection
        await page.goto(req.url, wait_until='networkidle', timeout=30000)
        
        # Hard wait to allow SPAs and bot-protection redirects (like Cloudflare) to settle
        await asyncio.sleep(2.0)
        
        # Ensure we start at the top for consistency in tests
        try:
            await page.evaluate("window.scrollTo(0, 0)")
        except:
            pass
            
        return await extract_page_state(page, req.session_id)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

class InteractRequest(BaseModel):
    action: str
    element_id: int | None = None
    text: str | None = None
    expected_text: str | None = None
    session_id: str
    tab_id: str = "default"

class HumanoidInteraction:
    """Enhanced helper to simulate human-like mouse and keyboard behavior natively."""
    
    @staticmethod
    def _bezier_curve(p0, p1, p2, p3, t):
        """Cubic Bezier curve formula."""
        return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

    @staticmethod
    async def move_mouse_human(page, target_x, target_y):
        # Get current mouse position (approximate or last known)
        # Since we can't easily get current mouse pos from playwright, we assume a starting point 
        # or use a small random offset from the center of the screen if it's the first move.
        start_x, start_y = random.randint(0, 1920), random.randint(0, 1080) 
        
        # Create two random control points to make the path curved
        cp1_x = start_x + random.uniform(-100, 100)
        cp1_y = start_y + random.uniform(-100, 100)
        cp2_x = target_x + random.uniform(-100, 100)
        cp2_y = target_y + random.uniform(-100, 100)
        
        steps = random.randint(15, 30)
        for i in range(steps + 1):
            t = i / steps
            x = HumanoidInteraction._bezier_curve(start_x, cp1_x, cp2_x, target_x, t)
            y = HumanoidInteraction._bezier_curve(start_y, cp1_y, cp2_y, target_y, t)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.005, 0.02))
        
        await asyncio.sleep(random.uniform(0.1, 0.3))

    @staticmethod
    async def type_human(page, selector, text):
        await page.locator(selector).click() # Focus first
        for char in text:
            await page.keyboard.type(char)
            await asyncio.sleep(random.uniform(0.02, 0.1))

    @staticmethod
    async def scroll_human(page, delta):
        chunks = random.randint(3, 7)
        for _ in range(chunks):
            step = delta // chunks + random.randint(-20, 20)
            await page.mouse.wheel(0, step)
            await asyncio.sleep(random.uniform(0.1, 0.3))

@app.post("/interact")
async def interact(req: InteractRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        await page.bring_to_front()
        
        print(f"DEBUG: [INTERACT] Action={req.action}, ID={req.element_id}, Text={req.text}", flush=True)
        
        if req.action == 'wait':
            wait_time = float(req.text) if req.text else 5.0
            print(f"DEBUG: [INTERACT] Waiting for {wait_time} seconds...", flush=True)
            await asyncio.sleep(wait_time)
            
        elif req.action == 'scroll_down':
            await HumanoidInteraction.scroll_human(page, 800)
            
        elif req.action == 'scroll_up':
            await HumanoidInteraction.scroll_human(page, -800)
            
        elif req.element_id is not None:
            selector = f'[aeon-id="{req.element_id}"]'
            locator = page.locator(selector).first
            
            if not await locator.count():
                return {"status": "error", "msg": f"Element ID {req.element_id} not found in DOM"}

            # Let scroll settle before grabbing bounding box
            await locator.scroll_into_view_if_needed()
            await asyncio.sleep(0.5) 
            
            box = await locator.bounding_box()
            if not box:
                return {"status": "error", "msg": f"Element ID {req.element_id} has no valid bounding box (might be invisible)."}

            # Slight jitter avoids pixel-perfect automated clicking
            target_x = box['x'] + box['width']/2 + random.uniform(-2, 2)
            target_y = box['y'] + box['height']/2 + random.uniform(-2, 2)

            if req.action == 'click':
                if req.expected_text:
                    try:
                        actual_text = await locator.inner_text()
                        alt_text = await locator.evaluate("(el) => el.value || el.getAttribute('aria-label') || el.name || el.title || ''")
                        img_alt = await locator.evaluate("(el) => { let img = el.querySelector('img'); return img ? img.alt : ''; }")
                        combined_text = (actual_text + " " + alt_text + " " + img_alt).replace('\n', ' ').strip().lower()
                        expected_clean = req.expected_text.replace('\n', ' ').strip().lower()
                        if expected_clean not in combined_text and combined_text not in expected_clean:
                            return {"status": "error", "msg": f"Safety Lock Triggered: Expected '{req.expected_text}', found '{combined_text}'"}
                    except Exception:
                        pass
                
                await HumanoidInteraction.move_mouse_human(page, target_x, target_y)
                # Playwright's native click with delay is extremely reliable and mimics human click speed
                await locator.click(delay=random.randint(50, 150))
                print(f"DEBUG: [INTERACT] Human-clicked ID {req.element_id}", flush=True)
                
            elif req.action == 'type':
                await HumanoidInteraction.move_mouse_human(page, target_x, target_y)
                await HumanoidInteraction.type_human(page, selector, req.text)
                print(f"DEBUG: [INTERACT] Human-typed '{req.text}' into ID {req.element_id}", flush=True)
                
            elif req.action == 'hover':
                await HumanoidInteraction.move_mouse_human(page, target_x, target_y)
                print(f"DEBUG: [INTERACT] Human-hovered ID {req.element_id}", flush=True)
                
            elif req.action == 'enter':
                await locator.press('Enter')
                
            elif req.action == 'select':
                if req.text:
                    try:
                        await locator.select_option(label=req.text)
                    except Exception:
                        await locator.select_option(value=req.text)
                else:
                    await locator.select_option(index=0)
                print(f"DEBUG: [INTERACT] Selected '{req.text}' in ID {req.element_id}", flush=True)
        else:
            return {"status": "error", "msg": "Invalid action or missing element_id"}
            
        await asyncio.sleep(random.uniform(2.0, 4.0))
        return await extract_page_state(page, req.session_id)
    except Exception as e:
        print(f"DEBUG: [INTERACT-ERROR] {e}", flush=True)
        return {"status": "error", "msg": str(e)}

class CloseTabRequest(BaseModel):
    session_id: str
    tab_id: str

@app.post("/close_tab")
async def close_tab(req: CloseTabRequest):
    global contexts, tabs
    key = f"{req.session_id}_{req.tab_id}"
    if key in tabs:
        try:
            await tabs[key].close()
        except:
            pass
        del tabs[key]
    
    remaining = sum(1 for k in tabs.keys() if k.startswith(f"{req.session_id}_"))
    
    # Clean up context if no tabs left for this session
    if remaining == 0:
        if req.session_id in contexts:
            try:
                await contexts[req.session_id].close()
            except:
                pass
            del contexts[req.session_id]
            
    return {"status": "ok", "remaining_tabs": remaining}

class CloseSessionRequest(BaseModel):
    session_id: str

@app.post("/close_session")
async def close_session(req: CloseSessionRequest):
    global contexts, tabs
    keys_to_delete = [k for k in tabs.keys() if k.startswith(f"{req.session_id}_")]
    for k in keys_to_delete:
        try:
            await tabs[k].close()
        except:
            pass
        del tabs[k]
    
    if req.session_id in contexts:
        try:
            await contexts[req.session_id].close()
        except:
            pass
        del contexts[req.session_id]
        
    return {"status": "ok"}

async def extract_page_state(page, session_id=None, fast_mode=False):
    # Wait for the body to be visible to avoid blank white pages
    try:
        await page.wait_for_selector("body", state="visible", timeout=3000)
    except Exception:
        pass

    # Optimization: In fast_mode, we significantly reduce quality and skip the clean screenshot
    quality = 60 if fast_mode else 95
    
    # Only take clean screenshot if not in fast_mode
    clean_bytes = None
    if not fast_mode:
        clean_bytes = await page.screenshot(type='jpeg', quality=quality)
    
    # Inject Set-of-Mark (SOM) script using a Python RAW string (r''') to prevent \n evaluation
    elements = await page.evaluate(r'''() => {
        let id = 0;
        let elements = [];
        document.querySelectorAll('.aeon-box').forEach(e => e.remove());
        
        // Recursive function to find all interactables, including those in Shadow DOMs
        const allInteractables = [];
        function findInteractables(root) {
            const selectors = 'a, button, input, textarea, select, summary, [role="button"], [role="link"], [role="menuitem"], iframe';
            const found = root.querySelectorAll(selectors);
            found.forEach(el => allInteractables.push(el));
            
            // Recurse into shadow roots
            const allElements = root.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.shadowRoot) {
                    findInteractables(el.shadowRoot);
                }
            });
        }
        
        findInteractables(document.body);
        console.log(`SOM: Found ${allInteractables.length} potential interactables (including Shadow DOM)`);
        
        allInteractables.forEach((el, index) => {
            // STABLE ID: Use the index in the DOM list so IDs don't shift when elements move off-screen
            let stableId = index + 1;
            el.setAttribute('aeon-id', stableId);
            
            let rect = el.getBoundingClientRect();
            let style = window.getComputedStyle(el);
            
            // Ensure element is visible AND within horizontal/vertical bounds
            let isVisible = (rect.width > 0 && rect.height > 0) && style.visibility !== 'hidden' && style.opacity !== '0';
            let inViewport = rect.top < window.innerHeight && rect.bottom > 0 && rect.left < window.innerWidth && rect.right > 0;
            
            if(!isVisible || !inViewport) {
                // Log only a sample of rejected elements to avoid flooding the console
                if (index % 5 === 0) {
                    console.log(`SOM: Rejecting <${el.tagName.toLowerCase()}> (ID:${stableId}): visible=${isVisible}, viewport=${inViewport} (top:${rect.top})`);
                }
            }
            
            if(isVisible && inViewport) {
                // We use the stableId for the label and the elements list
                let boxLeft = Math.max(0, rect.left);
                let boxTop = Math.max(0, rect.top);
                let boxWidth = Math.min(rect.width, window.innerWidth - boxLeft);
                let boxHeight = Math.min(rect.height, window.innerHeight - boxTop);
                
                let box = document.createElement('div');
                box.className = 'aeon-box';
                box.style.position = 'absolute';
                box.style.left = (boxLeft + window.scrollX) + 'px';
                box.style.top = (boxTop + window.scrollY) + 'px';
                box.style.width = boxWidth + 'px';
                box.style.height = boxHeight + 'px';
                box.style.border = '2px solid red';
                box.style.boxSizing = 'border-box';
                box.style.zIndex = 99999;
                box.style.pointerEvents = 'none';
                
                let label = document.createElement('span');
                label.innerText = stableId;
                label.style.backgroundColor = 'red';
                label.style.color = 'white';
                label.style.fontSize = '14px';
                label.style.fontWeight = 'bold';
                label.style.padding = '1px 3px';
                label.style.position = 'absolute';
                label.style.top = '-2px';
                label.style.left = '-2px';
                
                box.appendChild(label);
                document.body.appendChild(box);
                
                // INTELLIGENT CONTEXT EXTRACTION
                let text = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || el.name || '').replace(/\n/g, ' ').trim();
                
                if (el.tagName.toLowerCase() === 'iframe') {
                    text = `IFrame: ${el.title || el.name || el.id || el.src || 'Unknown Frame'}`;
                }
                else if (el.tagName.toLowerCase() === 'select') {
                    let opts = Array.from(el.options).map(o => o.text).join(' | ');
                    text = `Selected: ${el.options[el.selectedIndex]?.text || 'None'} [Options: ${opts}]`;
                }

                // Deep scan: Many e-commerce sites wrap images in <a> tags with no innerText. Grab the img alt!
                let img = el.querySelector('img');
                if (img && img.alt) {
                    let altText = img.alt.replace(/\n/g, ' ').trim();
                    text = text ? text + ' - ' + altText : altText;
                }
                
                // Escalation: If text is still generic, grab parent product card context
                let genericWords = ['click here', 'view', 'buy', 'view products', 'shop now'];
                if (text.length < 15 || genericWords.includes(text.toLowerCase())) {
                    try {
                        let parent = el.closest('article, li, .product, .card, .grid-item') || el.parentElement;
                        if (parent) {
                            let pImg = parent.querySelector('img');
                            let parentText = (parent.innerText || '').replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
                            let extraCtx = (pImg && pImg.alt) ? pImg.alt : parentText;
                            if (extraCtx) {
                                text = text ? text + " [Context: " + extraCtx.substring(0, 60) + "]" : extraCtx.substring(0, 60);
                            }
                        }
                    } catch(e) {}
                }
                
                text = text.substring(0, 100);
                elements.push({id: stableId, tag: el.tagName.toLowerCase(), text: text});
            }
        });
        return elements;
    }''')
    
    overlay_bytes = await page.screenshot(type='jpeg', quality=quality)
    
    # Extract visible markdown text and title/url
    try:
        markdown = await page.evaluate('() => document.body.innerText')
    except Exception:
        markdown = ""
        
    try:
        title = await page.title()
    except Exception:
        title = "Unknown"
        
    # Cleanup boxes after screenshot to not pollute the real DOM state long-term
    await page.evaluate('() => document.querySelectorAll(".aeon-box").forEach(e => e.remove())')
    
    res = {
        "status": "success",
        "clean_b64": base64.b64encode(clean_bytes).decode() if clean_bytes else None,
        "overlay_b64": base64.b64encode(overlay_bytes).decode(),
        "elements": elements,
        "markdown": markdown[:4000], # Truncate to save context window
        "title": title,
        "url": page.url
    }
    if session_id:
        res["open_tabs"] = [k[len(session_id)+1:] for k in tabs.keys() if k.startswith(f"{session_id}_")]
    return res

@app.get("/health")
async def health():
    return {"status": "ok"}
