import os, time, openai, pathlib, sys, json, re, requests, google.auth, google.auth.transport.requests
from datetime import datetime
from typing import Dict, Optional
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens
from .prompts import COMPRESS_ACTION_LOG_PROMPT, ANALYZE_INTERRUPTION_PROMPT, SUMMARIZE_TEXT_PROMPT, COMPRESS_MEMORIES_PROMPT

C_YEL, C_RES = '\033[93m', '\033[0m'

class VertexAIClient:
    def __init__(self, p_id, m_id):
        self.p_id, self.m_id = p_id, m_id
        try: self.creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        except Exception as e: sys.exit(f'\n{C_YEL}Err Auth: {e}\nRun: gcloud auth application-default login{C_RES}')
        self.chat = type('Chat', (), {'completions': type('Comps', (), {'create': self._crt})()})()
        
    def _tok(self):
        if not self.creds.valid: self.creds.refresh(google.auth.transport.requests.Request())
        return self.creds.token

    def _crt(self, model, messages, temperature=0.7, response_format=None, stream=False):
        sys_p = next((m['content'] for m in messages if m['role']=='system'), None)
        cts = [{'role': 'user' if m['role']=='user' else 'model', 'parts': [{'text': m['content']}]} for m in messages if m['role']!='system']
        d = {'contents': cts, 'generationConfig': {'temperature': temperature, 'maxOutputTokens': 8192}, 'safetySettings': [{'category': f'HARM_CATEGORY_{c}', 'threshold': 'OFF'} for c in ['HATE_SPEECH','DANGEROUS_CONTENT']]}
        if response_format and response_format.get('type')=='json_object': d['generationConfig']['responseMimeType'] = 'application/json'
        if sys_p: d['systemInstruction'] = {'parts': [{'text': sys_p}]}
        r = requests.post(f'https://aiplatform.googleapis.com/v1/projects/{self.p_id}/locations/global/publishers/google/models/{self.m_id}:generateContent', headers={'Authorization': f'Bearer {self._tok()}', 'Content-Type': 'application/json'}, json=d)
        if r.status_code != 200: raise Exception(f'Vertex API Err {r.status_code}: {r.text}')
        t = r.json().get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')
        M = lambda **kw: type('M',(),kw)()
        return [M(choices=[M(delta=M(content=t))])] if stream else M(choices=[M(message=M(content=t))], usage=M(completion_tokens=len(t)//4))

class LLMClient:
    def __init__(self, s_cfg: dict, w_cfg: dict):
        self.log, self.dbg_path, self.iter = get_logger(), None, 0
        if not s_cfg or not w_cfg: raise ValueError("Strong and weak configs required.")
        self.p_prov, self.p_mod, self.u_mod = s_cfg['provider'], s_cfg['model'], w_cfg['model']
        self.p_cli, self.u_cli = self._mk_cli(s_cfg), self._mk_cli(w_cfg)
        self.ctx_lim = min(s_cfg.get('context_limit', 128000), w_cfg.get('context_limit', 128000))

    def _mk_cli(self, c: dict):
        p = c['provider']
        if p == 'local': return openai.OpenAI(base_url='http://localhost:8013/v1', api_key='ollama')
        if p in ('llamacpp', 'vllm'): return openai.OpenAI(base_url=c['base_url'], api_key='none')
        if p == 'vertex': return VertexAIClient(c['project_id'], c['model'])
        return openai.OpenAI(api_key=open(pathlib.Path.home()/c['api_key_file']).readline().strip(), base_url=c['base_url'])

    def set_debug_path(self, p: pathlib.Path): self.dbg_path = p
    def set_iteration(self, i: int): self.iter = i
    def _log_to_debug(self, t, m, p, r): pass

    def _cln_json(self, c: str) -> str:
        if not c: return "{}"
        c = re.sub(r'</?think>|<think>.*?</think>|```json\s*|```\s*', '', c, flags=re.S).strip()
        b, s, e, in_s, esc = 0, -1, -1, False, False
        for i, ch in enumerate(c):
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_s = not in_s
            elif not in_s:
                if ch == '{':
                    if s == -1: s = i
                    b += 1
                elif ch == '}':
                    b -= 1
                    if b == 0 and s != -1: e = i + 1; break
        if s != -1 and e != -1: return c[s:e]
        m = re.search(r'\{.*\}', c, re.S)
        return m.group(0) if m else "{}"

    def _find_jend(self, r: str) -> int:
        s = r.find('{')
        if s == -1: return -1
        b, in_s, esc = 0, False, False
        for i in range(s, len(r)):
            ch = r[i]
            if esc: esc = False
            elif ch == '\\' and in_s: esc = True
            elif ch == '"': in_s = not in_s
            elif not in_s:
                if ch == '{': b += 1
                elif ch == '}':
                    b -= 1
                    if b == 0: return i + 1
        return -1

    def _xtr_blks(self, r: str, end: int) -> dict:
        b, rem = {}, r[end:] if end > 0 else r
        for m in re.finditer(r'^[^\S\n]*-*\s*BEGIN[\s_]+BLOCK[\s_]*(\d+)\s*-*\s*$\n?(.*?)^[^\S\n]*-*\s*END[\s_]+BLOCK[\s_]*\1\s*-*\s*$', rem, re.S | re.M):
            c = m.group(2); b[f'BLOCK_{m.group(1)}'] = c[:-1] if c.endswith('\n') else c
        if not b:
            for m in re.finditer(r'<{2,4}(BLOCK_[A-Za-z0-9_]+)>{2,4}\n?(.*?)<{2,4}END_\1>{2,4}', rem, re.S):
                c = m.group(2); b[m.group(1)] = c[:-1] if c.endswith('\n') else c
        return b

    def _xtr_inln(self, v: str):
        m = re.search(r'(?:^|\n)\s*-*\s*BEGIN[\s_]+BLOCK[\s_]*\d+\s*-*\s*\n(.*?)\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*(?:\n|$)', v, re.S)
        if m: return m.group(1)
        m = re.search(r'<{2,4}BLOCK_\w+>{2,4}\n?(.*?)\n?<{2,4}END_BLOCK_\w+>{2,4}', v, re.S)
        if m: return m.group(1)[:-1] if m.group(1).endswith('\n') else m.group(1)
        m = re.match(r'^[_<]{1,4}BLOCK[\s_]*\d+[_>]{1,4}\s*\n(.*)', v, re.S)
        if m: return re.sub(r'\n\s*<{2,4}END_BLOCK_\w+>{2,4}\s*$', '', re.sub(r'\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*$', '', m.group(1)))
        return None

    def _sub_blks(self, o, b: dict, mb: list = None):
        mb = mb if mb is not None else []
        if isinstance(o, dict): return {k: self._sub_blks(v, b, mb) for k, v in o.items()}
        if isinstance(o, list): return [self._sub_blks(i, b, mb) for i in o]
        if isinstance(o, str):
            s = o.strip()
            m = re.match(r'^(?:__BLOCK[_\s]*(\d+)__|<{2,4}BLOCK[_\s]*(\d+)>{2,4})$', s)
            if m:
                k = f"BLOCK_{m.group(1) or m.group(2)}"
                if k in b: return b[k]
                if k not in mb: mb.append(k)
                return o
            if '\n' in o and 'BLOCK' in o:
                x = self._xtr_inln(o)
                if x is not None: return x
        return o

    def _u_call(self, p: str, fmt=None, temp=0.7) -> Optional[str]:
        kw = {'model': self.u_mod, 'messages': [{"role": "user", "content": p}], 'temperature': temp}
        if fmt: kw['response_format'] = fmt
        try: return self.u_cli.chat.completions.create(**kw).choices[0].message.content.strip()
        except Exception as e: self.log.warning(f"Util Call Err: {e}"); return None

    def _recov_blk(self, k: str, j: dict, p: str) -> Optional[str]:
        r = self._u_call(f"{p}\n\n=== RECOVERY ===\nIntent: '{j.get('intent', 'Unk')}'. Forgot {k}.\nOutput EXACT raw code for {k}. NO JSON. NO MARKDOWN.", temp=0.1)
        if r and r.startswith("```") and r.endswith("```"):
            l = r.split('\n')
            return '\n'.join(l[1:-1]) if len(l) >= 3 else r.strip('`')
        return r

    def _rep_json(self, raw: str, e: str) -> Optional[str]:
        r = self._u_call(f"Strict JSON repair. Fix escaped quotes/newlines. Output ONLY valid JSON.\n{raw}", temp=0.0)
        return self._cln_json(r) if r else None

    def _chk_conn(self, e) -> bool:
        self.log.warning(f"Conn err: {e}. Recovery mode (10m)...")
        st, d = time.time(), 1
        while time.time() - st < 600:
            time.sleep(d)
            try:
                if isinstance(self.p_cli, VertexAIClient): self.p_cli.chat.completions.create(model=self.p_mod, messages=[{"role":"user","content":"hi"}], temperature=0)
                else: self.p_cli.models.list()
                return True
            except: d = min(d * 2, 60)
        return False

    def compress_action_log(self, t: str) -> str: return self._u_call(COMPRESS_ACTION_LOG_PROMPT.format(log=t)) or t
    def compress_memories(self, t: str) -> Dict:
        r = self._u_call(COMPRESS_MEMORIES_PROMPT.format(memories=t), {"type": "json_object"})
        return json.loads(self._cln_json(r)) if r else {}
    def analyze_interruption(self, o, i) -> Dict:
        r = self._u_call(ANALYZE_INTERRUPTION_PROMPT.format(obj=o, inp=i), {"type": "json_object"})
        return json.loads(self._cln_json(r)) if r else {"classification": "ADVICE", "updated_text": i, "reasoning": "Fail"}
    def reason(self, p: str) -> str:
        try: return self.p_cli.chat.completions.create(model=self.p_mod, messages=[{"role":"user","content":p}]).choices[0].message.content
        except Exception as e: return f"Err: {e}"
    def summarize_text(self, t: str, q: str) -> str: return self._u_call(SUMMARIZE_TEXT_PROMPT.format(query=q, text=t)) or "Fail"

    def _truncate_with_tail(self, t: str, hl: int = 500, tl: int = 1000) -> str:
        return t if len(t) <= (hl + tl) else f"{t[:hl]}\n... [TRUNC {len(t)-(hl+tl)}] ...\n{t[-tl:]}"

    def get_primary_agent_response(self, p: str, retries: int = 3, diag: Optional[str] = None) -> str:
        cp, l_err = p, None
        for att in range(retries):
            try:
                st = time.time()
                if not isinstance(self.p_cli, VertexAIClient):
                    strm = self.p_cli.chat.completions.create(model=self.p_mod, messages=[{"role": "user", "content": cp}], temperature=0.2, stream=True)
                    ft, chunks = None, []
                    for c in strm:
                        if ft is None: ft = time.time()
                        if hasattr(c, 'choices') and c.choices and hasattr(c.choices[0].delta, 'content') and c.choices[0].delta.content:
                            chunks.append(c.choices[0].delta.content)
                    raw, gt = "".join(chunks), time.time() - (ft or st)
                    tok = estimate_tokens(raw)
                    print(f"\033[96m[Perf] {self.p_mod}: {tok/gt if gt>0 else 0:.2f} t/s (TTFT: {(ft-st) if ft else 0:.2f}s | {tok}t in {gt:.2f}s)\033[0m")
                else:
                    resp = self.p_cli.chat.completions.create(model=self.p_mod, messages=[{"role": "user", "content": cp}], temperature=0.2)
                    raw, el = resp.choices[0].message.content, time.time() - st
                    tok = getattr(resp.usage, 'completion_tokens', estimate_tokens(raw))
                    print(f"\033[96m[Perf] {self.p_mod}: {tok/el if el>0 else 0:.2f} t/s (TTFT: N/A | {tok}t in {el:.2f}s)\033[0m")

                if self.dbg_path: print(f"{C_YEL}[LLM RAW]\n{raw}{C_RES}")
                je = self._find_jend(raw)
                blks = self._xtr_blks(raw, je)
                js = raw[:je] if je > 0 else raw
                cln = self._cln_json(js)

                try:
                    pd = json.loads(cln)
                    if not pd or 'actions' not in pd: raise ValueError("JSON missing 'actions'")
                    mb = []
                    pd = self._sub_blks(pd, blks, mb)
                    
                    if mb:
                        if self.dbg_path: print(f"{C_YEL}[LLM] Rec missing: {mb}{C_RES}")
                        for m in mb:
                            rt = self._recov_blk(m, pd, cp)
                            if not rt: raise ValueError(f"Rec fail: {m}")
                            blks[m] = rt
                        mb.clear()
                        pd = self._sub_blks(pd, blks, mb)
                        if mb: raise ValueError(f"Still missing: {mb}")
                    
                    if blks and self.dbg_path: print(f"{C_YEL}[LLM] Sub {len(blks)} blk(s){C_RES}")
                    return json.dumps(pd)
                except (json.JSONDecodeError, ValueError) as e:
                    l_err = str(e)
                    self.log.warning(f"Pri Att {att+1}/{retries} fail: {l_err}")
                    if (isinstance(e, json.JSONDecodeError) or "Expecting" in l_err or "Unterminated" in l_err) and "Empty" not in l_err:
                        if self.dbg_path: print(f"{C_YEL}[LLM] JSON Repair via {self.u_mod}...{C_RES}")
                        rep = self._rep_json(js, l_err)
                        if rep:
                            try:
                                pd = self._sub_blks(json.loads(rep), blks)
                                if 'actions' in pd: return json.dumps(pd)
                            except Exception as re_err: self.log.warning(f"Repair fail: {re_err}")

                if diag: print(f"\n{C_YEL}--- ROT DIAG ({att+1}) ---\n{diag}\n----------------{C_RES}\n")
                if att < retries - 1: cp = p + f"\n\n** RETRY - INVALID PREV **\nErr: {l_err}\nRaw: {raw[:300]}...\nReturn VALID JSON. Use block format (--- BEGIN BLOCK_N ---) for multi-line strings."
                
            except (openai.APIConnectionError, openai.InternalServerError, requests.exceptions.ConnectionError) as e:
                if self._chk_conn(e): continue
                raise
            except Exception as e:
                self.log.error(f"Pri err: {e}")
                if "401" in str(e) and isinstance(self.p_cli, VertexAIClient):
                    try: self.p_cli.creds.refresh(google.auth.transport.requests.Request())
                    except: pass
                l_err = str(e)
                if att < retries - 1: time.sleep(2); continue
                raise

        self.log.error(f"Pri fail 3/3. Lst: {l_err}")
        raise RuntimeError(f"Pri failed. Lst: {l_err}")