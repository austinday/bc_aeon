import asyncio, base64, sys, random
from fastapi import FastAPI
from pydantic import BaseModel
from camoufox.async_api import AsyncCamoufox

app = FastAPI()
brw = None
ctxs = {} # session_id -> context
tabs = {} # session_id_tab_id -> page
pop_cnts = {}
pop_lck = asyncio.Lock()

@app.on_event("startup")
async def startup():
    global brw
    try: brw = await AsyncCamoufox(headless=False, geoip=False, args=['--width=1920', '--height=1080', '--window-position=0,0']).__aenter__()
    except Exception as e: print(f"Err: {e}", file=sys.stderr)

@app.on_event("shutdown")
async def shutdown():
    if brw: await brw.__aexit__(None, None, None)

async def on_popup(p):
    if p in tabs.values(): return
    sid = next((k for k, v in ctxs.items() if v == p.context), None)
    if not sid: return
    async with pop_lck:
        pop_cnts[sid] = pop_cnts.get(sid, 0) + 1
        tabs[f"{sid}_popup_{pop_cnts[sid]}"] = p

async def get_page(sid: str, tid: str):
    if sid not in ctxs:
        ctxs[sid] = await brw.new_context(device_scale_factor=2)
        ctxs[sid].on("page", lambda p: asyncio.create_task(on_popup(p)))
    k = f"{sid}_{tid}"
    if k not in tabs:
        p = await ctxs[sid].new_page()
        await p.set_viewport_size({"width": 1920, "height": 1080})
        await p.add_init_script("window.addEventListener('DOMContentLoaded', ()=>{ let s=document.createElement('style'); s.textContent='html,body{max-width:100vw!important;overflow-x:hidden!important;margin:0!important;padding:0!important;}'; document.head.appendChild(s);});")
        tabs[k] = p
        for pk in [x for x, v in tabs.items() if v == p and 'popup_' in x]: del tabs[pk]
    return tabs[k]

class GotoReq(BaseModel): url: str; session_id: str; tab_id: str = "default"
class TabReq(BaseModel): session_id: str; tab_id: str
class SessReq(BaseModel): session_id: str
class IntReq(BaseModel):
    action: str; element_id: int | None = None; text: str | None = None; expected_text: str | None = None
    session_id: str; tab_id: str = "default"

class Human:
    @staticmethod
    async def move(p, tx, ty):
        sx, sy = random.randint(0, 100), random.randint(0, 100)
        cx1, cy1 = sx+(tx-sx)*random.uniform(.1,.4), sy+(ty-sy)*random.uniform(.1,.4)
        cx2, cy2 = sx+(tx-sx)*random.uniform(.6,.9), sy+(ty-sy)*random.uniform(.6,.9)
        stps = random.randint(10, 25)
        for i in range(stps + 1):
            t = i / stps
            await p.mouse.move((1-t)**3*sx + 3*(1-t)**2*t*cx1 + 3*(1-t)*t**2*cx2 + t**3*tx, (1-t)**3*sy + 3*(1-t)**2*t*cy1 + 3*(1-t)*t**2*cy2 + t**3*ty)
            await asyncio.sleep(random.uniform(.005, .015))
    
    @staticmethod
    async def type(p, sel, txt):
        await p.locator(sel).click()
        for c in txt: await p.keyboard.type(c); await asyncio.sleep(random.uniform(.05, .2))
        
    @staticmethod
    async def scroll(p, d):
        c = random.randint(3, 7)
        for _ in range(c): await p.mouse.wheel(0, d//c + random.randint(-50, 50)); await asyncio.sleep(random.uniform(.2, .6))

@app.post("/switch_tab")
async def switch_tab(r: TabReq):
    try:
        p = await get_page(r.session_id, r.tab_id); await p.bring_to_front(); await asyncio.sleep(1)
        return await ext_state(p, r.session_id)
    except Exception as e: return {"status": "error", "msg": str(e)}

@app.post("/navigate")
async def navigate(r: GotoReq):
    try:
        p = await get_page(r.session_id, r.tab_id); await p.bring_to_front()
        await p.goto(r.url, wait_until='domcontentloaded', timeout=15000)
        await p.evaluate("window.scrollTo(0, 0)"); await asyncio.sleep(1.5)
        return await ext_state(p, r.session_id)
    except Exception as e: return {"status": "error", "msg": str(e)}

@app.post("/interact")
async def interact(r: IntReq):
    try:
        p = await get_page(r.session_id, r.tab_id); await p.bring_to_front()
        if r.action == 'scroll_down': await Human.scroll(p, 800)
        elif r.action == 'scroll_up': await Human.scroll(p, -800)
        elif r.element_id is not None:
            sel = f'[aeon-id="{r.element_id}"]'; loc = p.locator(sel).first
            if not await loc.count(): return {"status": "error", "msg": "Element not found"}
            await loc.scroll_into_view_if_needed(); b = await loc.bounding_box()
            tx, ty = (b['x'] + b['width']/2, b['y'] + b['height']/2) if b else (0, 0)
            
            if r.action == 'click':
                if r.expected_text:
                    txt = await loc.inner_text()
                    alt = await loc.evaluate("(e)=>e.value||e.getAttribute('aria-label')||e.name||e.title||''")
                    ialt = await loc.evaluate("(e)=>{let i=e.querySelector('img');return i?i.alt:'';}")
                    cmb = f"{txt} {alt} {ialt}".replace('\n', ' ').strip().lower()
                    exp = r.expected_text.replace('\n', ' ').strip().lower()
                    if exp not in cmb and cmb not in exp: return {"status": "error", "msg": f"Safety lock: found '{cmb}'"}
                await Human.move(p, tx, ty); await asyncio.sleep(random.uniform(.1, .3)); await loc.click(delay=random.randint(50, 150))
            elif r.action == 'type': await Human.move(p, tx, ty); await loc.click(); await Human.type(p, sel, r.text)
            elif r.action == 'hover': await Human.move(p, tx, ty); await asyncio.sleep(random.uniform(.3, .8))
            elif r.action == 'enter': await loc.press('Enter')
            elif r.action == 'select':
                try: await loc.select_option(label=r.text) if r.text else await loc.select_option(index=0)
                except: await loc.select_option(value=r.text)
        else: return {"status": "error", "msg": "Invalid action"}
        await asyncio.sleep(random.uniform(2.0, 4.0))
        return await ext_state(p, r.session_id)
    except Exception as e: return {"status": "error", "msg": str(e)}

async def clean_ctx(sid):
    if not any(k.startswith(f"{sid}_") for k in tabs) and sid in ctxs:
        try: await ctxs[sid].close()
        except: pass
        del ctxs[sid]

@app.post("/close_tab")
async def close_tab(r: TabReq):
    k = f"{r.session_id}_{r.tab_id}"
    if k in tabs:
        try: await tabs[k].close()
        except: pass
        del tabs[k]
    await clean_ctx(r.session_id)
    return {"status": "ok", "remaining_tabs": sum(1 for x in tabs if x.startswith(f"{r.session_id}_"))}

@app.post("/close_session")
async def close_session(r: SessReq):
    for k in [x for x in tabs if x.startswith(f"{r.session_id}_")]:
        try: await tabs[k].close()
        except: pass
        del tabs[k]
    await clean_ctx(r.session_id)
    return {"status": "ok"}

async def ext_state(p, sid=None):
    cb = await p.screenshot(type='jpeg', quality=95)
    els = await p.evaluate(r'''()=>{
        let els=[]; document.querySelectorAll('.a-box').forEach(e=>e.remove());
        document.querySelectorAll('a,button,input,textarea,select,summary,[role="button"],[role="link"],[role="menuitem"]').forEach((e,i)=>{
            let id=i+1; e.setAttribute('aeon-id',id); let r=e.getBoundingClientRect(), s=window.getComputedStyle(e);
            if(r.width>0&&r.height>0&&s.visibility!=='hidden'&&s.opacity!=='0'&&r.top<window.innerHeight&&r.bottom>0&&r.left<window.innerWidth&&r.right>0){
                let bl=Math.max(0,r.left), bt=Math.max(0,r.top), bw=Math.min(r.width,window.innerWidth-bl), bh=Math.min(r.height,window.innerHeight-bt);
                let b=document.createElement('div'); b.className='a-box';
                b.style.cssText=`position:absolute;left:${bl+window.scrollX}px;top:${bt+window.scrollY}px;width:${bw}px;height:${bh}px;border:2px solid red;box-sizing:border-box;z-index:99999;pointer-events:none;`;
                let l=document.createElement('span'); l.innerText=id; l.style.cssText='background:red;color:white;font-size:14px;font-weight:bold;padding:1px 3px;position:absolute;top:-2px;left:-2px;';
                b.appendChild(l); document.body.appendChild(b);
                
                let t=(e.innerText||e.value||e.getAttribute('aria-label')||e.title||e.name||'').replace(/\n/g,' ').trim();
                if(e.tagName.toLowerCase()==='select') t=`Sel: ${e.options[e.selectedIndex]?.text||'None'} [Opts: ${Array.from(e.options).map(o=>o.text).join('|')}]`;
                let img=e.querySelector('img'); if(img&&img.alt) t=t?t+' - '+img.alt.replace(/\n/g,' ').trim():img.alt.replace(/\n/g,' ').trim();
                
                if(t.length<15||['click here','view','buy','view products','shop now'].includes(t.toLowerCase())){
                    try{ let pr=e.closest('article,li,.product,.card,.grid-item')||e.parentElement;
                        if(pr){ let pi=pr.querySelector('img'), pt=(pr.innerText||'').replace(/\s+/g,' ').trim(), ctx=(pi&&pi.alt)?pi.alt:pt;
                            if(ctx) t=t?t+" [Ctx: "+ctx.substring(0,60)+"]":ctx.substring(0,60);
                        }}catch(err){}
                }
                els.push({id:id, tag:e.tagName.toLowerCase(), text:t.substring(0,100)});
            }
        }); return els;
    }''')
    ob = await p.screenshot(type='jpeg', quality=95)
    md = await p.evaluate('()=>document.body.innerText')
    await p.evaluate('()=>document.querySelectorAll(".a-box").forEach(e=>e.remove())')
    
    res = {
        "status": "success", "clean_b64": base64.b64encode(cb).decode(),
        "overlay_b64": base64.b64encode(ob).decode(), "elements": els, "markdown": md[:4000]
    }
    if sid: res["open_tabs"] = [k[len(sid)+1:] for k in tabs if k.startswith(f"{sid}_")]
    return res

@app.get("/health")
async def health(): return {"status": "ok"}