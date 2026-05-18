import os,argparse,json,time,sys,subprocess as sp,requests as rq,fcntl,signal,atexit
from pathlib import Path

PID=os.getpid()
TMP="/tmp"
LCK=f"{TMP}/aeon_runtime.lock"
RST_ST=f"{TMP}/aeon_restart_state_{PID}.json"
RST_BAK=f"{TMP}/aeon_restart_backup_{PID}.tar.gz"
START_LCK=f"{TMP}/aeon_brain_startup.lock"
REG=f"{TMP}/aeon_model_registry.json"
REG_LCK=f"{TMP}/aeon_model_registry.lock"

CLOUD_MODELS=[
 {'model':'gemini-3.1-pro-preview','provider':'vertex','project_id':'trout-cricket-9761108088181001','context_limit':2000000},
 {'model':'gemini-3.1-pro-preview','provider':'vertex','project_id':'ai-ml-355015','context_limit':2000000},
 {'model':'grok-4.3-latest','provider':'grok','api_key_file':'grok_api_key.txt','base_url':'https://api.x.ai/v1','context_limit':128000},
 {'model':'gemini-3-pro-preview','provider':'gemini','api_key_file':'gemini_api_key.txt','base_url':'https://generativelanguage.googleapis.com/v1beta/openai/','context_limit':1000000},
 {'model':'gemini-flash-latest','provider':'gemini','api_key_file':'gemini_api_key.txt','base_url':'https://generativelanguage.googleapis.com/v1beta/openai/','context_limit':1000000}
]

LLAMACPP_MODELS=[
 {'model':'Qwen3.6-35B-A3B-Uncensored','family':'Qwen3.6','label':'Qwen3.6-35B-A3B-Uncensored | GPU0: 100%, GPU1: 0% | ~?? t/s | 256k ctx | Abliterated: Yes | Local/llama.cpp','provider':'llamacpp','base_url':'http://localhost:8009/v1','context_limit':262144,'container_name':'aeon_qwen36_35b','start_script':'start_qwen36_35b.sh','health_port':8009},
 {'model':'Gemma-4-31B-MTP-Q8_0','family':'Gemma-4','label':'Gemma-4-31B Native MTP Cluster | Symmetrical Dual 256k | ~100+ t/s | 256k ctx | Abliterated: Yes | Local/llama.cpp','provider':'llamacpp','base_url':'http://localhost:8013/v1','context_limit':262144,'container_name':'aeon_gemma_mtp_lb','additional_containers':['aeon_gemma4_mtp_node0','aeon_gemma4_mtp_node1'],'start_script':'start_gemma4_mtp.sh','health_port':8013},
 {'model':'Gemma-4-31B-NVFP4','family':'Gemma-4','label':'Gemma-4-31B NVFP4 Turbo | vLLM MTP | ~100+ t/s | 128k ctx | Abliterated: Yes | Local/vLLM','provider':'vllm','base_url':'http://localhost:8018/v1','context_limit':131072,'container_name':'aeon_gemma_vllm_lb','additional_containers':['gemma_node0','gemma_node1'],'start_script':'0_launch_gemma_nvfp4.sh','health_port':8018}
]

def is_container_running(n):
 try:return bool(sp.check_output(["docker","ps","-q","-f",f"name={n}"],stderr=sp.DEVNULL,text=True).strip())
 except:return False

def wait_for_service(n,p,ep="/api/tags",t=60):
 s=time.time()
 while time.time()-s<t:
  try:
   if rq.get(f"http://localhost:{p}{ep}",timeout=2).status_code==200:return True
  except:pass
  time.sleep(2)
 return False

def start_local_brain_services():
 if is_container_running("aeon_brain_node"):return True
 e=os.environ.copy();e["AEON_HOME"]=e.get("AEON_HOME",os.path.expanduser("~/.aeon"))
 sp.run(["bash",str(Path(__file__).parent/"scripts"/"start_brain.sh")],check=True,env=e)
 return wait_for_service("Aeon Brain (Ollama)",8000,ep="/api/tags",t=120)

def warm_up_models(ms):
 for m in dict.fromkeys(ms or []):
  try:rq.post("http://localhost:8000/api/generate",json={"model":m,"prompt":"hello","options":{"num_predict":1}},timeout=300)
  except:pass

def _pid_ok(p):
 if p==PID:return True
 try:
  os.kill(p,0)
  try:
   if open(f"/proc/{p}/stat").read().split()[2]=='Z':return False
  except:return False
  try:
   c=open(f"/proc/{p}/cmdline").read().replace('\x00',' ').strip().lower()
   if "aeon.main" not in c and "sub_agent_wrapper" not in c and not c.endswith("aeon"):return False
  except:return False
  return True
 except OSError:return False

def cleanup_transient_tools():
 sp.run("docker ps -a -q --filter 'name=aeon_research' | xargs -r docker rm -f",shell=True,stderr=sp.DEVNULL,timeout=5)
 def _cln(rp,lp,cn,cb=None):
  try:
   with open(lp,'w') as fd:
    fcntl.flock(fd,fcntl.LOCK_EX)
    d=json.load(open(rp)) if os.path.exists(rp) else []
    ps=d.get("pids",d) if isinstance(d,dict) else d
    o=[p for p in (ps if isinstance(ps,list) else []) if isinstance(p,int) and p!=PID and _pid_ok(p)]
    if cb:cb()
    if not o:sp.run(["docker","rm","-f",cn],stderr=sp.DEVNULL)
  except:pass
 _cln(f"{TMP}/aeon_comfyui_registry.json",f"{TMP}/aeon_comfyui_registry.lock","aeon_comfyui")
 _cln(f"{TMP}/aeon_vision_vllm_registry.json",f"{TMP}/aeon_vision_vllm_registry.lock","aeon_qwen36_vl")
 def _cls_br():
  try:rq.post("http://localhost:8030/close_session",json={"session_id":str(PID)},timeout=2)
  except:pass
 _cln(f"{TMP}/aeon_browser_registry.json",f"{TMP}/aeon_browser_registry.lock","aeon_browser",_cls_br)

def is_llamacpp_model(c):return c and c.get('provider') in ['llamacpp','vllm']
def get_llamacpp_config(m):return next((x for x in LLAMACPP_MODELS if x['model']==m),None)

def start_llamacpp_server(c):
 cn,p,s=c['container_name'],c['health_port'],c['start_script']
 if is_container_running(cn):
  try:
   if rq.get(f'http://localhost:{p}/health',timeout=5).status_code==200:return True
  except:pass
 scr=Path(__file__).parent/'scripts'/s
 if not scr.exists():return False
 e=os.environ.copy();e["AEON_HOME"]=e.get("AEON_HOME",os.path.expanduser("~/.aeon"))
 if sp.run(['bash',str(scr)],env=e).returncode!=0:return False
 return wait_for_service(c['model'],p,ep="/health",t=900)

def stop_llamacpp_server(c):
 for cn in [c['container_name']]+c.get('additional_containers',[]):
  try:sp.run(['docker','rm','-f',cn],capture_output=True,timeout=30)
  except:pass

def unload_local_brain():
 try:
  r=rq.get("http://localhost:8000/api/ps",timeout=3)
  if r.status_code==200:
   for m in r.json().get('models',[]):rq.post("http://localhost:8000/api/generate",json={"model":m['name'],"keep_alive":0},timeout=10)
 except:pass

def _mod_reg(ms,add=True):
 if not ms:return
 u=[]
 with open(REG_LCK,'w') as fd:
  fcntl.flock(fd,fcntl.LOCK_EX)
  try:r=json.load(open(REG)) if os.path.exists(REG) else {}
  except:r={}
  cr={m:[p for p in ps if _pid_ok(p)] for m,ps in r.items()}
  o=[m for m,ps in r.items() if not cr.get(m)]
  for k in o:cr.pop(k,None)
  u.extend(o)
  for m in ms:
   cr.setdefault(m,[])
   if add and PID not in cr[m]:cr[m].append(PID)
   elif not add and PID in cr[m]:
    cr[m].remove(PID)
    if not cr[m]:del cr[m];u.append(m)
  with open(REG,'w') as f:json.dump(cr,f)
 for m in set(u):
  lc=get_llamacpp_config(m)
  if lc:stop_llamacpp_server(lc)
  else:
   try:rq.post("http://localhost:8000/api/generate",json={"model":m,"keep_alive":0},timeout=15)
   except:pass

def register_models_for_agent(ms):_mod_reg(ms,True)
def unregister_models_for_agent(ms):_mod_reg(ms,False)

def cleanup_ghost_llamacpp_containers():
 try:
  r=sp.run(["docker","ps","--format","{{.Names}}"],capture_output=True,text=True,check=True)
  reg=json.load(open(REG)) if os.path.exists(REG) else {}
  for c in r.stdout.splitlines():
   if not c.startswith("aeon_"):continue
   cfg=next((x for x in LLAMACPP_MODELS if x['container_name']==c),None)
   if not cfg:continue
   m=cfg['model']
   if not any(_pid_ok(p) for p in reg.get(m,[])):sp.run(["docker","rm","-f",c],stdout=sp.DEVNULL,stderr=sp.DEVNULL)
 except:pass

def get_ollama_models():
 try:
  r=rq.get("http://localhost:8000/api/tags",timeout=1)
  if r.status_code==200:return sorted([m['name'] for m in r.json().get('models',[])])
 except:pass
 return []

def terminate_all_sub_agents():
 d=Path("aeon_output")
 if not d.exists():return
 for f in d.rglob("pid.txt"):
  if "sub_agents" in f.parts:
   try:
    p=int(f.read_text().strip())
    os.kill(p,signal.SIGKILL)
    try:os.waitpid(p,0)
    except:
     for _ in range(10):
      try:os.kill(p,0);time.sleep(0.1)
      except:break
   except:pass
   try:(f.parent/"status.txt").write_text("KILLED")
   except:pass

class SessionManager:
 def __init__(self):
  self.r_lck=self.s_lck=self.cln=None
  self.osig=self.osig_t=None
  self.ms=[]
  self.lcpp_cfgs=[]
 def enter(self,s_cfg=None,w_cfg=None,skip_warmup=False):
  lm=list(dict.fromkeys([c['model'] for c in [s_cfg,w_cfg] if c and c.get('provider')=='local']))
  self.ms=list(lm)
  self.lcpp_cfgs=[c for c in [s_cfg,w_cfg] if is_llamacpp_model(c)]
  if lm:
   self.s_lck=open(START_LCK,'w+')
   try:
    fcntl.flock(self.s_lck,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if start_local_brain_services() and not skip_warmup:warm_up_models(lm)
    fcntl.flock(self.s_lck,fcntl.LOCK_SH)
   except BlockingIOError:fcntl.flock(self.s_lck,fcntl.LOCK_SH)
  for c in self.lcpp_cfgs:
   m=c['model'];register_models_for_agent([m]);self.ms.append(m);start_llamacpp_server(c)
  if lm:register_models_for_agent(lm)
  self.r_lck=open(LCK,'w+');fcntl.flock(self.r_lck,fcntl.LOCK_SH)
  self.osig_t=signal.signal(signal.SIGTERM,self._sig)
  atexit.register(self.exit)
 def _sig(self,s,f):self.exit();sys.exit(0)
 def exit(self):
  if self.cln:return
  self.cln=True
  try:
   osigi=signal.getsignal(signal.SIGINT)
   signal.signal(signal.SIGINT,signal.SIG_IGN)
  except:osigi=None
  try:
   terminate_all_sub_agents();cleanup_transient_tools()
   if self.ms:unregister_models_for_agent(self.ms)
   if self.r_lck:
    try:fcntl.flock(self.r_lck,fcntl.LOCK_UN);self.r_lck.close()
    except:pass
   if self.s_lck:
    try:self.s_lck.close()
    except:pass
   if self.osig_t:signal.signal(signal.SIGTERM,self.osig_t)
  finally:
   if osigi:signal.signal(signal.SIGINT,osigi)