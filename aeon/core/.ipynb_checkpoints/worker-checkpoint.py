import json,re,time,sys,os,uuid
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List,Any,Dict,Callable,Optional
from .llm import LLMClient
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens as est_tok
from .prompts import CORE_DIRECTIVES,DOCKER_DIRECTIVES,IMPORTANT_REMINDERS,PRIMARY_AGENT_INSTRUCTIONS,TOOLS_SECTION,OBJECTIVE_SECTION
CR,CY,CC,CG,CX,CB='\033[91m','\033[93m','\033[96m','\033[95m','\033[0m','\033[96m'

class Worker:
    def __init__(self,llm:LLMClient,tools:List[Any]=None,pr:Callable=print,dbg:bool=False,dbg_log:Optional[str]=None):
        self.llm,self.dbg_log,self.pr,self.dbg,self.log=llm,dbg_log,pr,dbg,get_logger()
        self.tools={t.name:t for t in tools} if tools else {}
        from aeon.core.prompts.manager import ensure_prompt_files
        from aeon.tools.categories import get_all_category_paths
        ensure_prompt_files(list(self.tools.keys()),get_all_category_paths())
        self._dbg_init=False
        if self.dbg: self._init_dbg()
        self.mems,self.ofiles,self.of_mtime,self.p_cache={},{},{},{}
        self.act_log,self._rec_cmds,self._rec_outs,self.of_lru=[],[],[],[]
        self.exp_cats,self.notif_subs=set(),set()
        self.plan,self.last_obs,self.act_log_sum="No plan formulated yet.","None.",""
        self.pend_state,self.obj,self.model_name=None,None,None
        self.rec_intents=deque(maxlen=3)
        self.prev_toks,self.eff_iters,self.MAX_REP,self.REP_THRESH,self.max_hist=0,0,5,2,30000
        self.iid=str(uuid.uuid4())[:8]
        self.b_dirs,self.dkr_dirs,self.imp_rems=CORE_DIRECTIVES,DOCKER_DIRECTIVES,IMPORTANT_REMINDERS

    def _init_dbg(self):
        if self._dbg_init: return
        self.dpath=Path.home()/f"aeon_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.pr(f"{CY}Debug logging enabled: {self.dpath}{CX}"); self._dbg_init=True

    def _sync_ofiles(self,max_len=250000):
        from aeon.tools.analyzers import FileAnalyzer
        for p in list(self.ofiles.keys()):
            if not os.path.exists(p):
                self.ofiles.pop(p,None); self.of_mtime.pop(p,None); self.log.info(f"Removed deleted: {p}"); continue
            try:
                mt=os.path.getmtime(p)
                if self.of_mtime.get(p)==mt and len(self.ofiles.get(p,""))<=max_len: continue
                res=FileAnalyzer(p).analyze(); st=res.get('summary_type','')
                if st=='opaque_binary': c=f"File '{p}' is binary. Use a script."
                elif st=='error': c=f"Error reading: {res.get('error_message','Unknown')}"
                elif st in ('empty_file','empty'): c='(empty file)'
                elif st=='full_content':
                    r=res.get('content','')
                    c=json.dumps(r,indent=2) if isinstance(r,(dict,list)) else str(r)
                else:
                    pts=[f'[File Summary: {st}]']
                    for k,v in res.items():
                        if k not in ('file_name','file_size_bytes','summary_type'): pts.append(f"{k}: {json.dumps(v,indent=2,default=str) if isinstance(v,(dict,list)) else v}")
                    c='\n'.join(pts)
                if len(c)>max_len: c=f"File '{p}' too large ({len(c):,} chars). Limit {max_len:,}. Use a script."
                if self.ofiles.get(p)!=c: self.ofiles[p]=c
                self.of_mtime[p]=mt
            except Exception as e: self.log.error(f"Error syncing {p}: {e}")

    def register_tools(self,tl:List[Any]):
        for t in tl: t.worker=self; self.tools[t.name]=t

    def update_open_file(self,p:str,c:str):
        ap=os.path.abspath(p); self.ofiles[ap]=c
        if ap in self.of_lru: self.of_lru.remove(ap)
        self.of_lru.append(ap)
        try: self.of_mtime[ap]=os.path.getmtime(ap)
        except OSError: pass

    def close_file(self,p:str)->bool:
        ap=os.path.abspath(p); t=ap if ap in self.ofiles else (p if p in self.ofiles else None)
        if t:
            del self.ofiles[t]
            if t in self.of_lru: self.of_lru.remove(t)
            return True
        return False

    def is_file_open(self,p:str)->bool: return os.path.abspath(p) in self.ofiles or p in self.ofiles

    def _get_act_dirs(self)->str:
        from aeon.tools.categories import TOP_LEVEL_TOOLS,get_all_categorized_tools as gact,get_tools_in_category as gtic
        from aeon.core.prompts.manager import load_cat_prompt as lcp,load_tool_prompt as ltp
        ad=[]; ct=gact(); atn=set(TOP_LEVEL_TOOLS)|{n for n in self.tools if n not in ct}
        for cp in self.exp_cats: atn.update(gtic(cp))
        for n in sorted(atn):
            if n not in self.p_cache: self.p_cache[n]=ltp(n)
            ad.extend([f"- {n}: {d}" for d in self.p_cache[n]])
        for cp in sorted(self.exp_cats):
            if cp not in self.p_cache: self.p_cache[cp]=lcp(cp)
            ad.extend([f"- {cp}: {d}" for d in self.p_cache[cp]])
        return "\n".join(ad)

    def _get_tools_desc(self)->str:
        from aeon.tools.categories import TOOL_CATEGORIES as TC,TOP_LEVEL_TOOLS as TLT,get_all_categorized_tools as gact
        ct=gact(); td=[f"- {n}: {t.description}" for n,t in self.tools.items() if n in TLT or n not in ct]
        res="\n\n".join(td); cl=self._rndr_cats(TC,'',0)
        if cl: res+='\n\n**TOOL CATEGORIES** (use expand_tool_category / collapse_tool_category to manage)\n'+'\n'.join(cl)
        return res

    def _rndr_cats(self,cats:dict,pp:str,d:int)->list:
        from aeon.tools.categories import count_tools_in_category as ctic
        ls=[]; ind='  '*d
        for n,c in cats.items():
            p=f'{pp}/{n}' if pp else n; exp=p in self.exp_cats; desc=c.get('description','')
            if exp:
                ls.append(f'{ind}[-] {n}: {desc}')
                ls.extend([f"{ind}  - {tn}: {self.tools[tn].description if tn in self.tools else '(not loaded)'}" for tn in c.get('tools',[])])
                if 'subcategories' in c: ls.extend(self._rndr_cats(c['subcategories'],p,d+1))
            else: tc=ctic(p); ls.append(f'{ind}[+] {n}: {desc} ({tc} tool{"s" if tc!=1 else ""})')
        return ls

    def _fmt_ofiles(self,max_len=250000)->str:
        self._sync_ofiles(max_len)
        if not self.ofiles: return "No files currently open."
        return "\n\n".join([f"--- FILE: {p} ---\n{c}\n--- END FILE: {p} ---" for p,c in self.ofiles.items()])

    def _fmt_mems(self)->str:
        if not self.mems: return "No memories recorded yet."
        return "\n".join([f"[{v.get('category','general')}] {k}: {v.get('value','')} (Saved: {v.get('timestamp','unknown')})" if isinstance(v,dict) else f"{k}: {v}" for k,v in self.mems.items()])

    def _trunc_out(self,t:str,max_c=50000)->str:
        if len(t)<=max_c: return t
        hb=max_c//4; tb=max_c-hb; omit=len(t)-max_c
        return t[:hb]+f"\n\n... [{omit:,} CHARS TRUNCATED] ...\n\n"+t[-tb:]

    def _fmt_alog(self,fl_only=False,pres="Low")->str:
        if not self.act_log and not self.pend_state: return "(No actions taken yet.)"
        fl="\n\n".join(self.act_log+([] if not self.pend_state else [f"[Iter {self.pend_state['iter']}]\n- Intent: {self.pend_state['intent']}\n- Actions: {', '.join(self.pend_state['actions'])}\n- Result: (Pending...)"]))
        if fl_only or est_tok(fl)<12000: return fl
        rc={"Low":12,"Moderate":8,"High":5,"CRITICAL":3}.get(pres,10)
        ls=[f"[HISTORICAL SUMMARY]\n{self.act_log_sum}"] if self.act_log_sum else []
        ls.extend(self.act_log[-rc:])
        if self.pend_state: ls.append(f"[Iter {self.pend_state['iter']}]\n- Intent: {self.pend_state['intent']}\n- Actions: {', '.join(self.pend_state['actions'])}\n- Result: (Pending...)")
        return "\n\n".join(ls)

    def _rst_state(self,iobs="Project started."):
        self.plan="Initial state. Need plan."; self.ofiles.clear(); self.mems.clear()
        self.last_obs=iobs; self.act_log.clear(); self.pend_state=None; self._rec_cmds.clear()
        self._rec_outs.clear(); self.exp_cats.clear(); self.notif_subs.clear(); self.eff_iters=0

    def serialize_state(self)->dict:
        return {'memories':dict(self.mems),'current_plan':self.plan,'action_log':list(self.act_log),'action_log_summary':self.act_log_sum,'objective':self.obj or '','expanded_categories':list(self.exp_cats),'notified_sub_agents':list(self.notif_subs),'instance_id':self.iid,'open_files_list':list(self.ofiles.keys()),'open_files_access_order':list(self.of_lru)}

    def restore_state(self,s:dict):
        self.mems,self.act_log,self.act_log_sum=s.get('memories',{}),s.get('action_log',[]),s.get('action_log_summary',"")
        self.exp_cats,self.notif_subs,self.of_lru=set(s.get('expanded_categories',[])),set(s.get('notified_sub_agents',[])),s.get('open_files_access_order',[])
        for p in s.get('open_files_list',[]): self.ofiles[p]="Restoring from state..."
        rsn=s.get('reason','code changes')
        self.act_log.append(f'[RESTART COMPLETED]\n- Reason: {rsn}\n- pip install: SUCCESS\n- Process relaunch: SUCCESS\n- State restore: SUCCESS (memories, action log preserved)\n- Result: Agent is NOW running the updated code. The restart is DONE.')
        self.plan=f'Restart completed successfully. The agent is now running with updated code ({rsn}). Next steps: verify the changes work as expected, then proceed with or complete the objective. DO NOT call restart_aeon again unless you make additional NEW code changes.'
        self.last_obs=f'=== RESTART COMPLETE ===\nThe agent process has been SUCCESSFULLY restarted. Details:\n- Code changes applied: {rsn}\n- The updated code is NOW ACTIVE in this running process.\n- All persistent memories and action history have been restored.\n\nCRITICAL: The restart is FINISHED. Do NOT call restart_aeon again.\nYour code changes are ALREADY LIVE. Proceed with verifying them or completing the task.'

    def _save_obj(self,obj:str):
        self.obj=obj
        try:
            with open(".previous_objective.txt","a",encoding="utf-8") as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] OBJECTIVE UPDATE:\n{obj}\n{'-'*40}\n")
        except Exception as e: self.log.error(f"Failed to save objective: {e}")

    def _bld_ctx(self,tls:str,sinf:str,mems:str,obj:str,ofs:str,adirs:str,alog:str,diag:str="")->str:
        rem=f"**IMPORTANT REMINDERS**\n{self.imp_rems}\n\n" if self.imp_rems.strip() else ""
        ds=f"\n**CONTEXT DIAGNOSTICS**\n{diag}\n" if diag else ""
        return f"{self.b_dirs}\n\n{self.dkr_dirs}\n\n**OPEN TOOL DIRECTIVES**\n{adirs or 'None'}\n\n{TOOLS_SECTION.format(tools=tls)}\n{rem}**PERSISTENT MEMORIES**\n{mems}\n\n**ATTEMPT LOG** (Historical record of intents and results)\n{alog}\n\n{sinf}\n{ds}\n**CURRENT PLAN**\n{self.plan}\n\n**OPEN FILES**\n===[ IN WORKING MEMORY ]===\n{ofs}\n===[ END OPEN FILES ]===\n\n**LAST STEP RESULT**\n{self.last_obs}\n\n{PRIMARY_AGENT_INSTRUCTIONS}\n\n{OBJECTIVE_SECTION.format(objective=obj)}"

    def _cln_json(self,r:str)->str:
        c=r.strip()
        if c.startswith("
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1