import os, argparse, json, time, sys, subprocess as sp, requests, fcntl, signal, atexit, shutil, tarfile
from pathlib import Path
from aeon.core.logger import get_logger
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

PID = os.getpid()
L_FILE, ST_LOCK = "/tmp/aeon_rt.lock", "/tmp/aeon_br_start.lock"
RST_ST, RST_BAK = f"/tmp/aeon_rst_{PID}.json", f"/tmp/aeon_bak_{PID}.tar.gz"
REG_F, REG_L = "/tmp/aeon_mod_reg.json", "/tmp/aeon_mod_reg.lock"

C_MODS = [
    {'model': 'gemini-3.1-pro-preview', 'provider': 'vertex', 'project_id': 'trout-cricket-9761108088181001', 'context_limit': 2000000},
    {'model': 'gemini-3.1-pro-preview', 'provider': 'vertex', 'project_id': 'ai-ml-355015', 'context_limit': 2000000},
    {'model': 'grok-4.3-latest', 'provider': 'grok', 'api_key_file': 'grok_api_key.txt', 'base_url': 'https://api.x.ai/v1', 'context_limit': 128000},
    {'model': 'gemini-3-pro-preview', 'provider': 'gemini', 'api_key_file': 'gemini_api_key.txt', 'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/', 'context_limit': 1000000},
    {'model': 'gemini-flash-latest', 'provider': 'gemini', 'api_key_file': 'gemini_api_key.txt', 'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/', 'context_limit': 1000000},
]

L_MODS = [
    {'model': 'Qwen3.6-35B-A3B-Uncensored', 'family': 'Qwen3.6', 'label': 'Qwen3.6-35B-A3B-Uncensored | GPU0:100%,GPU1:0% | Local/llama.cpp', 'provider': 'llamacpp', 'base_url': 'http://localhost:8009/v1', 'context_limit': 262144, 'cname': 'aeon_qwen36_35b', 'script': 'start_qwen36_35b.sh', 'port': 8009},
    {'model': 'Gemma-4-31B-MTP-Q8_0', 'family': 'Gemma-4', 'label': 'Gemma-4-31B Native MTP Cluster | Dual 256k | Local/llama.cpp', 'provider': 'llamacpp', 'base_url': 'http://localhost:8013/v1', 'context_limit': 262144, 'cname': 'aeon_gemma_mtp_lb', 'add_c': ['aeon_gemma4_mtp_node0', 'aeon_gemma4_mtp_node1'], 'script': 'start_gemma4_mtp.sh', 'port': 8013},
    {'model': 'Gemma-4-31B-NVFP4', 'family': 'Gemma-4', 'label': 'Gemma-4-31B NVFP4 Turbo | vLLM MTP | Local/vLLM', 'provider': 'vllm', 'base_url': 'http://localhost:8018/v1', 'context_limit': 131072, 'cname': 'aeon_gemma_vllm_lb', 'add_c': ['gemma_node0', 'gemma_node1'], 'script': '0_launch_gemma_nvfp4.sh', 'port': 8018},
]

def run_sh(c, **kw): return sp.run(c, shell=isinstance(c, str), **kw)
def is_run(n):
    try: return bool(run_sh(["docker","ps","-q","-f",f"name={n}"], capture_output=True, text=True).stdout.strip())
    except: return False
def wait_svc(n, p, ep="/api/tags", t=60):
    st = time.time()
    while time.time()-st < t:
        try:
            if requests.get(f"http://localhost:{p}{ep}", timeout=2).status_code==200: return True
        except: pass
        time.sleep(2)
    return False

def start_brain():
    if is_run("aeon_brain_node"): return True
    e = os.environ.copy()
    e["AEON_HOME"] = e.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    run_sh(["bash", str(Path(__file__).parent/"scripts"/"start_brain.sh")], check=True, env=e)
    return wait_svc("Brain", 8000, t=120)

def warm_mods(mods):
    for m in list(dict.fromkeys(mods or [])):
        try: requests.post("http://localhost:8000/api/generate", json={"model":m, "prompt":"hi", "options":{"num_predict":1}}, timeout=300)
        except: pass

def clean_trans():
    run_sh("docker ps -aq -f 'name=aeon_research' | xargs -r docker rm -f", stderr=sp.DEVNULL, timeout=5)
    def s_cln(rf, lf, cn, cb=None):
        try:
            with open(lf,'w') as fd:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if not os.path.exists(rf): return
                d = json.load(open(rf))
                pids, alv = d.get("pids",d) if isinstance(d,dict) else d, []
                if isinstance(pids,list):
                    for p in pids:
                        if not isinstance(p,int) or p==PID: continue
                        try:
                            os.kill(p,0)
                            if "aeon" in open(f"/proc/{p}/cmdline").read().replace('\x00',' ').lower(): alv.append(p)
                        except: pass
                if cb: cb()
                if not alv: run_sh(["docker","rm","-f",cn], stderr=sp.DEVNULL)
        except: pass
    s_cln("/tmp/aeon_comfyui_reg.json", "/tmp/aeon_comfyui_reg.lock", "aeon_comfyui")
    s_cln("/tmp/aeon_vis_vllm_reg.json", "/tmp/aeon_vis_vllm_reg.lock", "aeon_qwen36_vl")
    s_cln("/tmp/aeon_browser_reg.json", "/tmp/aeon_browser_reg.lock", "aeon_browser", lambda: requests.post("http://localhost:8030/close_session", json={"session_id":str(PID)}, timeout=2))

def is_llcpp(c): return c and c.get('provider') in ['llamacpp','vllm']
def get_llcpp(m): return next((c for c in L_MODS if c['model']==m), None)

def start_llcpp(c):
    cn, p = c['cname'], c['port']
    if is_run(cn):
        try:
            if requests.get(f'http://localhost:{p}/health', timeout=5).status_code==200: return True
        except: pass
    scr = Path(__file__).parent/'scripts'/c['script']
    if not scr.exists(): return False
    e = os.environ.copy()
    e["AEON_HOME"] = e.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    if run_sh(['bash', str(scr)], env=e).returncode!=0: return False
    return wait_svc(c['model'], p, "/health", 900)

def pid_ok(p):
    if p==PID: return True
    try:
        os.kill(p,0)
        if open(f"/proc/{p}/stat").read().split()[2]=='Z': return False
        if "aeon" not in open(f"/proc/{p}/cmdline").read().replace('\x00',' ').lower(): return False
        return True
    except: return False

def cln_ghosts():
    try:
        out = run_sh(["docker","ps","--format","{{.Names}}"], capture_output=True, text=True, check=True).stdout.splitlines()
        r = json.load(open(REG_F)) if os.path.exists(REG_F) else {}
        for c in out:
            if not c.startswith("aeon_"): continue
            cfg = next((x for x in L_MODS if x['cname']==c), None)
            if not cfg or not any(pid_ok(p) for p in r.get(cfg['model'],[])): run_sh(["docker","rm","-f",c], stderr=sp.DEVNULL, stdout=sp.DEVNULL)
    except: pass

def mod_reg(mods, add=True):
    if not mods: return
    unl = []
    with open(REG_L,'w') as lck:
        fcntl.flock(lck, fcntl.LOCK_EX)
        try: r = json.load(open(REG_F)) if os.path.exists(REG_F) else {}
        except: r = {}
        c_r, o_m = {}, []
        for m, pids in r.items():
            alv = [p for p in pids if pid_ok(p)]
            if alv: c_r[m] = alv
            else: o_m.append(m)
        if not add: unl.extend(o_m)
        r = c_r
        for m in mods:
            if add:
                if m not in r: r[m]=[]
                if PID not in r[m]: r[m].append(PID)
            else:
                if m in r and PID in r[m]:
                    r[m].remove(PID)
                    if not r[m]:
                        del r[m]
                        if m not in unl: unl.append(m)
        json.dump(r, open(REG_F,'w'))
    for m in (o_m if add else set(unl)):
        c = get_llcpp(m)
        if c: 
            for cn in [c['cname']] + c.get('add_c',[]): run_sh(['docker','rm','-f',cn], capture_output=True, timeout=30)
        else:
            try: requests.post("http://localhost:8000/api/generate", json={"model":m, "keep_alive":0}, timeout=15)
            except: pass

def get_oll():
    try:
        r = requests.get("http://localhost:8000/api/tags", timeout=1)
        if r.status_code==200: return sorted([m['name'] for m in r.json().get('models',[])])
    except: pass
    return []

def b_menu(l_mods):
    e = [{'l': '--- Local ---', 'h': 1}] + [{'model': m, 'provider': 'local', 'l': f'{m} | Local/Ollama'} for m in l_mods]
    lf = None
    for m in L_MODS:
        if lf is not None and m.get('family','Other') != lf: e.append({'l': '', 'h': 1})
        lf = m.get('family','Other')
        d = dict(m); d['l'] = m['label']; e.append(d)
    e += [{'l': '', 'h': 1}, {'l': '--- API ---', 'h': 1}]
    vm = []
    for c in C_MODS:
        d = dict(c)
        if c.get('provider')=='vertex': d['l'] = f"Vertex - {c['model']}"; vm.append(d)
        else: d['l'] = f"{c['model']} | API/Cloud"; e.append(d)
    if vm: e += [{'l': '', 'h': 1}, {'l': '--- Vertex ---', 'h': 1}] + vm
    return e

class Sess:
    def __init__(self): self.rl, self.sl, self.cd, self.osig, self.mu, self.lc = None, None, False, None, [], []
    def enter(self, sc=None, wc=None, no_warm=False):
        lms = list(dict.fromkeys([c['model'] for c in [sc, wc] if c and c.get('provider')=='local']))
        self.mu, self.lc = list(lms), [c for c in [sc,wc] if is_llcpp(c)]
        if lms:
            self.sl = open(ST_LOCK, 'w+')
            try:
                fcntl.flock(self.sl, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if start_brain() and not no_warm: warm_mods(lms)
            except BlockingIOError: pass
            finally: fcntl.flock(self.sl, fcntl.LOCK_SH)
        for c in self.lc:
            m = c['model']
            mod_reg([m], 1)
            if m not in self.mu: self.mu.append(m)
            start_llcpp(c)
        if lms: mod_reg(lms, 1)
        self.rl = open(L_FILE, 'w+')
        fcntl.flock(self.rl, fcntl.LOCK_SH)
        self.osig = signal.signal(signal.SIGTERM, lambda s,f: (self.exit(), sys.exit(0)))
        atexit.register(self.exit)
    def exit(self):
        if self.cd: return
        self.cd = True
        try: o_int = signal.getsignal(signal.SIGINT); signal.signal(signal.SIGINT, signal.SIG_IGN)
        except: o_int = None
        try:
            term_subs()
            clean_trans()
            if self.mu: mod_reg(self.mu, 0)
            for lck in [self.rl, self.sl]:
                if lck:
                    try: fcntl.flock(lck, fcntl.LOCK_UN) if lck==self.rl else None; lck.close()
                    except: pass
            if self.osig: signal.signal(signal.SIGTERM, self.osig)
        finally:
            if o_int: signal.signal(signal.SIGINT, o_int)

def term_subs():
    d = Path("aeon_output")
    if not d.exists(): return
    for pf in d.rglob("pid.txt"):
        if "sub_agents" in pf.parts:
            try:
                p = int(pf.read_text().strip())
                os.kill(p, 9)
                try: os.waitpid(p, 0)
                except:
                    for _ in range(10):
                        try: os.kill(p,0); time.sleep(0.1)
                        except: break
            except: pass
            if (sf := pf.parent / "status.txt").exists():
                try: sf.write_text("KILLED")
                except: pass

def _exec_rst(s, w=None):
    if not os.path.exists(RST_ST): return
    term_subs()
    try:
        st = json.load(open(RST_ST))
        obj, d = st.get('objective', ''), st.get('aeon_code_dir')
        def fail(m):
            if os.path.exists(RST_ST): os.remove(RST_ST)
            if w: w.last_observation, w.action_log = m, w.action_log + ['[RST FAIL]']
            return obj
        def rest_b(b):
            if b and os.path.exists(RST_BAK):
                if os.path.isdir(ad): shutil.rmtree(ad)
                tarfile.open(RST_BAK, 'r:gz').extractall(path=d)
                os.remove(RST_BAK)
        if not d or not os.path.isdir(d): return fail('Bad aeon_code_dir')
        ad, b_ex = os.path.join(d, 'aeon'), False
        if not os.path.isdir(ad): return fail('No aeon/ pkg dir')
        try:
            if os.path.exists(RST_BAK): os.remove(RST_BAK)
            tarfile.open(RST_BAK, 'w:gz').add(ad, arcname='aeon')
            b_ex = True
        except: pass
        for rt, ds, fs in os.walk(d):
            if '__pycache__' in ds: shutil.rmtree(os.path.join(rt, '__pycache__'), ignore_errors=True)
        if run_sh([sys.executable, '-m', 'pip', 'install', '.', '-q'], cwd=d).returncode != 0:
            rest_b(b_ex); return fail('Pip error')
        smp = os.path.join(ad, 'smoke_test.py')
        if os.path.exists(smp):
            res = run_sh([sys.executable, '-B', smp], capture_output=True, text=True, cwd=d)
            if res.returncode != 0:
                rest_b(b_ex)
                run_sh([sys.executable, '-m', 'pip', 'install', '.', '-q'], cwd=d)
                return fail(f'Smoke fail: {res.stdout}{res.stderr}')
        os.chdir(st.get('original_cwd', os.getcwd()))
        args = [sys.executable, '-B', '-m', 'aeon.main', '--resume', RST_ST, '--no-warmup'] + (['--debug'] if st.get('debug_mode') else []) + (['--model', st.get('model_name')] if st.get('model_name') else [])
        if b_ex and os.path.exists(RST_BAK): os.remove(RST_BAK)
        os.execv(sys.executable, args)
    except Exception as e:
        rest_b(b_ex)
        return fail(f'Exc: {e}')

def cli():
    cln_ghosts()
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true'); p.add_argument('--debug-log', type=str)
    p.add_argument('--strong', type=str, dest='model'); p.add_argument('--model', type=str)
    p.add_argument('--weak', type=str, help=argparse.SUPPRESS); p.add_argument('--start', type=str)
    p.add_argument('--no-warmup', action='store_true'); p.add_argument('--resume', type=str)
    a = p.parse_args()
    lms = get_oll() if is_run("aeon_brain_node") else (start_brain() and get_oll() or [])
    mn = b_menu(lms)
    mn_name = a.model or a.weak
    sc = next((x for x in mn if x.get('model')==mn_name), None) if mn_name else None
    if not sc and not mn_name:
        s = [x for x in mn if not x.get('h')]
        for x in mn: print(x['l'] and f" {x['l']}" or "") if x.get('h') else print(f" {s.index(x)+1:>2}. {x['l']}")
        while not sc:
            try: c = input(f'Select (1-{len(s)}): ')
            except: sys.exit(0)
            sc = s[int(c)-1] if c.isdigit() and 1<=int(c)<=len(s) else None
    if not sc: sys.exit(1)
    sess = Sess()
    try:
        sess.enter(sc, sc, a.no_warmup)
        llm = LLMClient(sc, sc)
        w = Worker(llm_client=llm, debug_mode=a.debug, debug_log_path=a.debug_log)
        w.model_name, w.model_config = sc['model'], sc
        w.register_tools(load_tools_from_directory("aeon.tools", dependencies={'llm_client': llm, 'worker': w}))
        if a.resume and os.path.exists(a.resume):
            try:
                st = json.load(open(a.resume)); os.remove(a.resume); w.restore_state(st)
                o = st.get('objective', '')
                while o: w.run(o); o = _exec_rst(sess, w)
            except: pass
            if os.path.exists(a.resume): os.remove(a.resume)
        o = a.start
        while o: w.run(o); o = _exec_rst(sess, w)
        while 1:
            try:
                o = input("> ")
                if o.strip() in ['exit', 'quit']: break
                while o: w.run(o); o = _exec_rst(sess, w)
            except: break
    finally: sess.exit()

if __name__ == "__main__": cli()