import os, json, base64, difflib, re
from .base import BaseTool
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE, TOOL_DESC_CLOSE_FILE, TOOL_DESC_WRITE_FILE, TOOL_DESC_STR_REPLACE
)
from .analyzers import FileAnalyzer

MAX_SIZE = 250000
FUZZ_THRESH = 0.6

class OpenFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name='open_file', description=TOOL_DESC_OPEN_FILE)
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path: return 'Err: file_path required.'
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path): return f'Err: File not found: {file_path}'
        if os.path.isdir(abs_path): return f'Err: {file_path} is directory. Use Project Tree.'
        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            return f"File '{file_path}' already open in working memory."

        try: res = FileAnalyzer(abs_path).analyze()
        except Exception as e: return f'Err analyzing: {type(e).__name__}: {e}'

        st = res.get('summary_type', '')
        if st == 'opaque_binary': return f"'{file_path}' is binary, cannot display."
        if st == 'error': return f"Err reading: {res.get('error_message', 'Unknown')}"
        if st in ('empty_file', 'empty'): c = '(empty file)'
        elif st == 'full_content':
            r = res.get('content', '')
            c = json.dumps(r, indent=2) if isinstance(r, (dict, list)) else str(r)
        else:
            p = [f'[File Summary: {st}]']
            for k, v in res.items():
                if k not in ('file_name', 'file_size_bytes', 'summary_type'):
                    p.append(f"{k}: {json.dumps(v, indent=2, default=str) if isinstance(v, (dict, list)) else v}")
            c = '\n'.join(p)

        if len(c) > MAX_SIZE:
            return f"'{file_path}' too large ({len(c):,} chars). Limit {MAX_SIZE:,} chars."

        self.worker.update_open_file(abs_path, c)
        disp = '\n'.join(f"{i+1}: {l}" for i, l in enumerate(c.splitlines()))
        return f"Opened '{file_path}' ({len(self.worker.open_files)} files open)\n\n---\n{disp}"


class CloseFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name='close_file', description=TOOL_DESC_CLOSE_FILE)
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path: return 'Err: file_path required.'
        return f"File '{file_path}' closed." if self.worker.close_file(file_path) else f"'{file_path}' was not open."


class StrReplaceTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name='str_replace', description=TOOL_DESC_STR_REPLACE)
        self.worker, self._fails = worker, {}

    def _norm_ws(self, t: str) -> str:
        return '\n'.join(l.rstrip() for l in t.splitlines())

    def _fuzzy(self, c: str, s: str):
        sl, cl = s.splitlines(True), c.splitlines(True)
        if not sl or not cl: return None, 0.0
        wsz, b_scr, b_st, b_end = len(sl), 0.0, -1, -1
        for d in (0, -1, 1, -2, 2):
            asz = wsz + d
            if not (1 <= asz <= len(cl)): continue
            for i in range(len(cl) - asz + 1):
                scr = difflib.SequenceMatcher(None, s, ''.join(cl[i:i+asz]), autojunk=False).ratio()
                if scr > b_scr: b_scr, b_st, b_end = scr, i, i+asz
        if b_scr >= FUZZ_THRESH and b_st >= 0:
            return ''.join(cl[b_st:b_end]), b_scr
        return None, b_scr

    def _replace(self, abs_path, path, c, old, new):
        so = old.strip()
        # 1. L-Syntax
        lm = re.match(r'^L(\d+)(?:-L(\d+))?$', so)
        if lm:
            try:
                sl = int(lm.group(1))
                el = int(lm.group(2)) if lm.group(2) else sl
                cls = c.splitlines(True)
                if not (1 <= sl <= len(cls)) or not (1 <= el <= len(cls)) or sl > el:
                    return c, None, f'Err: Invalid bounds L{sl}-L{el} (1-{len(cls)}).'
                nc = "".join(cls[:sl-1]) + new + "".join(cls[el:])
                if nc == c: return c, None, 'Warn: Identical content. No changes.'
                return nc, f'range (L{sl}-L{el})', None
            except Exception as e: return c, None, f'Err line range: {e}'

        # 2. Line-Num Stripping
        po = ''.join(re.sub(r'^(\d+):\s*', '', l) for l in old.splitlines(True))
        mtxt, mthd = None, 'exact'

        # 3. Exact Match
        if c.count(po) == 1: mtxt, _ = po, self._fails.pop(abs_path, None)
        elif c.count(po) > 1: return c, None, f'Err: matched {c.count(po)}x. Add context.'
        elif c.count(old) == 1: mtxt, _ = old, self._fails.pop(abs_path, None)
        elif c.count(old) > 1: return c, None, f'Err: matched {c.count(old)}x. Add context.'

        # 4. Whitespace-Normalized
        if not mtxt:
            nc_norm, sc_norm = self._norm_ws(c), self._norm_ws(po)
            if nc_norm.count(sc_norm) == 1:
                st_idx = nc_norm[:nc_norm.find(sc_norm)].count('\n')
                e_idx = st_idx + po.count('\n') + (0 if po.endswith('\n') else 1)
                mtxt = ''.join(c.splitlines(True)[st_idx:e_idx])
                mthd, _ = 'ws-norm', self._fails.pop(abs_path, None)
            elif nc_norm.count(sc_norm) > 1: return c, None, f'Err: matched {nc_norm.count(sc_norm)}x after ws-norm.'

        # 5. Fuzzy Match
        if not mtxt:
            fm, scr = self._fuzzy(c, old)
            if fm: mtxt, mthd, _ = fm, f'fuzzy ({scr:.1%})', self._fails.pop(abs_path, None)
            else:
                fc = self._fails.get(abs_path, 0) + 1
                self._fails[abs_path] = fc
                fl = old.split('\n')[0].strip()
                cls = c.splitlines()
                diag = ""
                for i, l in enumerate(cls):
                    if fl and fl[:30] in l:
                        st, en = max(0, i - 1), min(len(cls), i + 4)
                        diag = f"\nNear match:\n" + '\n'.join(f' L{st+j+1}: {cls[st+j]}' for j in range(en-st))
                        break
                if fc >= 3:
                    self._fails[abs_path] = 0
                    return c, None, f"Err: failed {fc}x. Score {scr:.1%}. STOP str_replace. Use write_file.\n{diag}"
                return c, None, f"Err: No match (try {fc}/3). Score {scr:.1%}. Verify text.\n{diag}"

        if mthd != 'exact' and c.count(mtxt) != 1:
            return c, None, f'Err: {mthd} match appears {c.count(mtxt)}x. Unsafe.'

        nc = c.replace(mtxt, new, 1)
        if nc == c: return c, None, 'Warn: Identical content. No changes.'
        return nc, mthd, None

    def execute(self, file_path: str, patch: str = None, old_str: str = None, new_str: str = '') -> str:
        if not file_path: return 'Err: file_path required.'
        if not patch and not old_str: return 'Err: patch or old_str required.'
        if new_str is None: new_str = ''
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path): return f'Err: Not found: {file_path}'
        if os.path.isdir(abs_path): return f'Err: {file_path} is dir.'

        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f: c = f.read()
        except Exception as e: return f'Err read: {type(e).__name__}: {e}'

        cur_c, used = c, []
        if patch:
            blks = re.findall(r'<<<<\s*SEARCH\n?(.*?)\n?====\n?(.*?)\n?>>>>\s*REPLACE', patch, re.DOTALL)
            if not blks: return "Err: Bad SEARCH/REPLACE blocks."
            for o, n in blks:
                if not o: continue
                nc, m, err = self._replace(abs_path, file_path, cur_c, o, n)
                if err and err.startswith('Err'): return err
                if m: used.append(m)
                cur_c = nc
        else:
            nc, m, err = self._replace(abs_path, file_path, cur_c, old_str, new_str)
            if err and err.startswith('Err'): return err
            if m: used.append(m)
            cur_c = nc

        if cur_c == c: return f"Warn: No changes to {file_path}."
        try:
            with open(abs_path, 'w', encoding='utf-8') as f: f.write(cur_c)
        except Exception as e: return f'Err write: {type(e).__name__}: {e}'

        if self.worker.is_file_open(abs_path) or self.worker.is_file_open(file_path):
            self.worker.update_open_file(abs_path, cur_c)

        return f"Applied {len(used) if patch else 1} block(s) to {file_path} (methods: {','.join(set(used)) or 'exact'})."


class WriteFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name='write_file', description=TOOL_DESC_WRITE_FILE)
        self.worker = worker

    def execute(self, file_path: str, content: str) -> str:
        if not file_path: return 'Err: file_path required.'
        if content is None: return 'Err: content required.'

        is_bin, c_dec = False, content
        if content.startswith('base64:'):
            try:
                c_dec = base64.b64decode(content[7:])
                try: c_dec = c_dec.decode('utf-8')
                except UnicodeDecodeError: is_bin = True
            except Exception as e: return f'Err b64 decode: {e}'

        abs_path = os.path.abspath(file_path)
        try:
            p_dir = os.path.dirname(abs_path)
            if p_dir: os.makedirs(p_dir, exist_ok=True)
            mode, kw = ('wb', {}) if is_bin else ('w', {'encoding': 'utf-8'})
            with open(abs_path, mode, **kw) as f: f.write(c_dec)

            self.worker.close_file(file_path)
            self.worker.close_file(abs_path)
            return f'Successfully wrote to {file_path}.'
        except PermissionError: return f'Err: Permission denied: {file_path}'
        except Exception as e: return f'Err write: {type(e).__name__}: {e}'