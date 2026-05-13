import asyncio
import base64
import sys
import random

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
        await page.bring_to_front()
        await asyncio.sleep(1)
        return await extract_page_state(page, req.session_id)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.post("/navigate")
async def navigate(req: GotoRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        await page.bring_to_front()  # CRITICAL: Bring the tab to the foreground
        await page.goto(req.url, wait_until='domcontentloaded', timeout=15000)
        
        # Ensure we start at the top for consistency in tests
        await page.evaluate("window.scrollTo(0, 0)")
        
        # Settle time to allow page to fully load and JS to execute
        await asyncio.sleep(1.5)
            
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

@app.post("/interact")
async def interact(req: InteractRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        await page.bring_to_front()
        
        print(f"DEBUG: [INTERACT] Action={req.action}, ID={req.element_id}, Text={req.text}", flush=True)
        
        if req.action == 'scroll_down':
            await page.mouse.wheel(0, 800)
        elif req.action == 'scroll_up':
            await page.mouse.wheel(0, -800)
        elif req.element_id is not None:
            selector = f'[aeon-id="{req.element_id}"]'
            locator = page.locator(selector).first
            
            # Verify element exists and is attached
            if not await locator.count():
                return {"status": "error", "msg": f"Element ID {req.element_id} not found in DOM"}

            if req.action == 'click':
                await locator.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
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
                
                await locator.click(delay=random.randint(50, 150))
                print(f"DEBUG: [INTERACT] Clicked ID {req.element_id}", flush=True)
            elif req.action == 'type':
                await locator.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
                await locator.fill(req.text)
                await asyncio.sleep(0.5)
                print(f"DEBUG: [INTERACT] Typed '{req.text}' into ID {req.element_id}", flush=True)
            elif req.action == 'hover':
                await locator.scroll_into_view_if_needed()
                box = await locator.bounding_box()
                if box:
                    await page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
                    await asyncio.sleep(0.5)
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
            
        # Settle time for JS and popups
        await asyncio.sleep(3)
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
    
    # Clean up context if no tabs left for this session
    if not any(k.startswith(f"{req.session_id}_") for k in tabs.keys()):
        if req.session_id in contexts:
            try:
                await contexts[req.session_id].close()
            except:
                pass
            del contexts[req.session_id]
            
    return {"status": "ok"}

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

async def extract_page_state(page, session_id=None):
    # Take clean screenshot
    clean_bytes = await page.screenshot(type='jpeg', quality=95)
    
    # Inject Set-of-Mark (SOM) script using a Python RAW string (r''') to prevent \n evaluation
    elements = await page.evaluate(r'''() => {
        let id = 0;
        let elements = [];
        document.querySelectorAll('.aeon-box').forEach(e => e.remove());
        
        const interactables = document.querySelectorAll('a, button, input, textarea, select, summary, [role="button"], [role="link"], [role="menuitem"]');
        console.log(`SOM: Found ${interactables.length} potential interactables`);
        
        interactables.forEach((el, index) => {
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
                let text = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || el.name || '').replace(/\\n/g, ' ').trim();
                
                if (el.tagName.toLowerCase() === 'select') {
                    let opts = Array.from(el.options).map(o => o.text).join(' | ');
                    text = `Selected: ${el.options[el.selectedIndex]?.text || 'None'} [Options: ${opts}]`;
                }

                // Deep scan: Many e-commerce sites wrap images in <a> tags with no innerText. Grab the img alt!
                let img = el.querySelector('img');
                if (img && img.alt) {
                    let altText = img.alt.replace(/\\n/g, ' ').trim();
                    text = text ? text + ' - ' + altText : altText;
                }
                
                // Escalation: If text is still generic, grab parent product card context
                let genericWords = ['click here', 'view', 'buy', 'view products', 'shop now'];
                if (text.length < 15 || genericWords.includes(text.toLowerCase())) {
                    try {
                        let parent = el.closest('article, li, .product, .card, .grid-item') || el.parentElement;
                        if (parent) {
                            let pImg = parent.querySelector('img');
                            let parentText = (parent.innerText || '').replace(/\\n/g, ' ').replace(/\\s+/g, ' ').trim();
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
    
    overlay_bytes = await page.screenshot(type='jpeg', quality=95)
    
    # Extract visible markdown text
    markdown = await page.evaluate('() => document.body.innerText')
    
    # Cleanup boxes after screenshot to not pollute the real DOM state long-term
    await page.evaluate('() => document.querySelectorAll(".aeon-box").forEach(e => e.remove())')
    
    res = {
        "status": "success",
        "clean_b64": base64.b64encode(clean_bytes).decode(),
        "overlay_b64": base64.b64encode(overlay_bytes).decode(),
        "elements": elements,
        "markdown": markdown[:4000] # Truncate to save context window
    }
    if session_id:
        # Return a list of all tab_ids associated with this session
        # Use slicing instead of split to correctly handle session_ids that contain underscores
        res["open_tabs"] = [k[len(session_id)+1:] for k in tabs.keys() if k.startswith(f"{session_id}_")]
    return res

@app.get("/health")
async def health():
    return {"status": "ok"}
