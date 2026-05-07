import asyncio
import base64
import sys

print("Loading imports...", flush=True)
from fastapi import FastAPI
from pydantic import BaseModel
from camoufox.async_api import AsyncCamoufox

print("Imports loaded. Initializing FastAPI...", flush=True)
app = FastAPI()
browser_instance = None
sessions = {}  # session_id_tab_id -> {"page": page}

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

async def get_or_create_session(session_id: str, tab_id: str):
    global browser_instance, sessions
    key = f"{session_id}_{tab_id}"
    if key not in sessions:
        page = await browser_instance.new_page()
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
        
        sessions[key] = {"page": page}
    return sessions[key]["page"]

class GotoRequest(BaseModel):
    url: str
    session_id: str
    tab_id: str = "default"

@app.post("/navigate")
async def navigate(req: GotoRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        await page.bring_to_front()  # CRITICAL: Bring the tab to the foreground
        await page.goto(req.url, wait_until='domcontentloaded', timeout=15000)
        await asyncio.sleep(2)
        return await extract_page_state(page)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

class InteractRequest(BaseModel):
    action: str
    element_id: int = None
    text: str = None
    expected_text: str = None
    session_id: str
    tab_id: str = "default"

@app.post("/interact")
async def interact(req: InteractRequest):
    try:
        page = await get_or_create_session(req.session_id, req.tab_id)
        await page.bring_to_front()  # CRITICAL: Bring the tab to the foreground before interacting
        
        if req.action == 'scroll_down':
            await page.mouse.wheel(0, 800)
        elif req.action == 'scroll_up':
            await page.mouse.wheel(0, -800)
        elif req.element_id is not None:
            selector = f'[aeon-id="{req.element_id}"]'
            locator = page.locator(selector).first
            
            if req.action == 'click':
                if req.expected_text:
                    try:
                        actual_text = await locator.inner_text()
                        alt_text = await locator.evaluate("(el) => el.value || el.getAttribute('aria-label') || el.name || el.title || ''")
                        img_alt = await locator.evaluate("(el) => { let img = el.querySelector('img'); return img ? img.alt : ''; }")
                        
                        combined_text = (actual_text + " " + alt_text + " " + img_alt).replace('\n', ' ').strip().lower()
                        expected_clean = req.expected_text.replace('\n', ' ').strip().lower()
                        
                        if expected_clean not in combined_text and combined_text not in expected_clean:
                            return {
                                "status": "error", 
                                "msg": f"Safety Lock Triggered: You clicked ID {req.element_id} expecting '{req.expected_text}', but the DOM text is '{combined_text}'. The VLM likely hallucinated the ID. Check the markdown elements list for the correct ID."
                            }
                    except Exception:
                        pass # Skip validation if element vanished
                        
                # Use JS click to completely bypass sticky/floating headers intercepting the event
                await locator.evaluate("node => node.click()")
            elif req.action == 'type':
                await locator.fill(req.text)
            elif req.action == 'enter':
                await locator.press('Enter')
        else:
            return {"status": "error", "msg": "Invalid action or missing element_id"}
            
        await asyncio.sleep(2)
        return await extract_page_state(page)
    except Exception as e:
        return {"status": "error", "msg": str(e)}

class CloseTabRequest(BaseModel):
    session_id: str
    tab_id: str

@app.post("/close_tab")
async def close_tab(req: CloseTabRequest):
    global sessions
    key = f"{req.session_id}_{req.tab_id}"
    if key in sessions:
        try:
            await sessions[key]["page"].close()
        except:
            pass
        del sessions[key]
    return {"status": "ok"}

class CloseSessionRequest(BaseModel):
    session_id: str

@app.post("/close_session")
async def close_session(req: CloseSessionRequest):
    global sessions
    keys_to_delete = [k for k in sessions.keys() if k.startswith(f"{req.session_id}_")]
    for k in keys_to_delete:
        try:
            await sessions[k]["page"].close()
        except:
            pass
        del sessions[k]
    return {"status": "ok"}

async def extract_page_state(page):
    # Take clean screenshot
    clean_bytes = await page.screenshot(type='jpeg', quality=80)
    
    # Inject Set-of-Mark (SOM) script using a Python RAW string (r''') to prevent \n evaluation
    elements = await page.evaluate(r'''() => {
        let id = 0;
        let elements = [];
        document.querySelectorAll('.aeon-box').forEach(e => e.remove());
        
        const interactables = document.querySelectorAll('a, button, input, textarea, select, summary, [role="button"], [role="link"], [role="menuitem"]');
        interactables.forEach(el => {
            let rect = el.getBoundingClientRect();
            
            // Ensure element is visible AND within horizontal/vertical bounds
            let isVisible = (rect.width > 0 && rect.height > 0) && window.getComputedStyle(el).visibility !== 'hidden' && window.getComputedStyle(el).opacity !== '0';
            let inViewport = rect.top < window.innerHeight && rect.bottom > 0 && rect.left < window.innerWidth && rect.right > 0;
            
            if(isVisible && inViewport) {
                id++;
                el.setAttribute('aeon-id', id);
                
                // Clamp bounding box to viewport to prevent scrollbar triggering or off-screen drawing
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
                label.innerText = id;
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
                elements.push({id: id, tag: el.tagName.toLowerCase(), text: text});
            }
        });
        return elements;
    }''')
    
    overlay_bytes = await page.screenshot(type='jpeg', quality=80)
    
    # Extract visible markdown text
    markdown = await page.evaluate('() => document.body.innerText')
    
    # Cleanup boxes after screenshot to not pollute the real DOM state long-term
    await page.evaluate('() => document.querySelectorAll(".aeon-box").forEach(e => e.remove())')
    
    return {
        "status": "success",
        "clean_b64": base64.b64encode(clean_bytes).decode(),
        "overlay_b64": base64.b64encode(overlay_bytes).decode(),
        "elements": elements,
        "markdown": markdown[:4000] # Truncate to save context window
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
