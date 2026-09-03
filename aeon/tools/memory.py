from datetime import datetime, timedelta, timezone
import re
from .base import BaseTool
from ..core.prompts import TOOL_DESC_MEMORIZE, TOOL_DESC_FORGET

class MemorizeTool(BaseTool):
    MAX_ITEMS = 80
    MAX_KEY_CHARS = 120
    MAX_VALUE_CHARS = 1000
    MAX_CATEGORY_CHARS = 80

    def __init__(self, worker):
        super().__init__(name="memorize", description=TOOL_DESC_MEMORIZE)
        self.worker = worker

    _SECRET_NAME_RE = re.compile(
        r"(?:password|passwd|secret|token|api[_ -]?key|private[_ -]?key|cookie|"
        r"session|credential|recovery|otp|2fa|authorization|bearer)", re.I
    )
    _SECRET_VALUE_RE = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
        r"(?:password|passwd|token|secret|api[_ -]?key|authorization)\s*[:=]\s*\S+|"
        r"bearer\s+[A-Za-z0-9._~+/-]{16,})",
        re.I,
    )

    @classmethod
    def secret_error(cls, key, value, category) -> str:
        name = f"{key} {category}"
        if cls._SECRET_NAME_RE.search(name) or cls._SECRET_VALUE_RE.search(str(value or "")):
            return (
                "Memory refused: secret-like credentials must stay in Nexus credential "
                "storage. Memorize only an opaque credential handle or the fact that a "
                "named integration is configured; never include the secret value."
            )
        return ""

    def execute(
        self,
        key: str,
        value: str,
        category: str = "general",
        scope: str = "task",
        ttl_hours: int = None,
    ) -> str:
        if not key or not value:
            return "Error: Both 'key' and 'value' are required."

        key = str(key).strip()
        value = str(value).strip()
        category = str(category or "general").strip()
        scope = str(scope or "task").strip().lower()
        if not key or not value:
            return "Error: key and value must contain non-whitespace text."
        if any(ord(char) < 32 for char in key + category):
            return "Error: key and category cannot contain control characters."
        if len(key) > self.MAX_KEY_CHARS:
            return f"Error: key exceeds {self.MAX_KEY_CHARS} characters."
        if len(value) > self.MAX_VALUE_CHARS:
            return (
                f"COMMAND BLOCKED: memory values must be concise (maximum "
                f"{self.MAX_VALUE_CHARS} characters). Store a durable artifact and memorize its path."
            )
        if len(category) > self.MAX_CATEGORY_CHARS:
            return f"Error: category exceeds {self.MAX_CATEGORY_CHARS} characters."
        if key not in self.worker.memories and len(self.worker.memories) >= self.MAX_ITEMS:
            return (
                f"COMMAND BLOCKED: memory is at its {self.MAX_ITEMS}-item limit. "
                "Forget stale task facts before adding another."
            )
        if scope not in {"task", "project", "preference"}:
            return "Error: scope must be one of: task, project, preference."
        secret_error = self.secret_error(key, value, category)
        if secret_error:
            return f"COMMAND BLOCKED: {secret_error}"
        now = datetime.now(timezone.utc)
        expires_at = None
        if ttl_hours is not None:
            try:
                ttl = int(ttl_hours)
            except (TypeError, ValueError):
                return "Error: ttl_hours must be a positive integer."
            if ttl <= 0 or ttl > 24 * 365:
                return "Error: ttl_hours must be between 1 and 8760."
            expires_at = (now + timedelta(hours=ttl)).isoformat()

        # Note when an existing key is being overwritten so the agent realizes it
        # replaced a fact rather than adding one.
        prev = self.worker.memories.get(key)
        prev_value = prev.get("value") if isinstance(prev, dict) else prev

        self.worker.memories[key] = {
            "value": value,
            "category": category,
            "scope": scope,
            "source": "model_observation",
            "timestamp": now.isoformat(),
            "expires_at": expires_at,
        }
        print(f"{self.C_CYAN}🧠 Memory saved: [{scope}/{category}] {key}{self.C_RESET}")

        if prev_value is not None and str(prev_value) != str(value):
            return f"Updated memory metadata/value for '{key}' [{scope}/{category}]."
        return f"Memorized non-secret fact '{key}' [{scope}/{category}]."

class ForgetTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name="forget", description=TOOL_DESC_FORGET)
        self.worker = worker

    def execute(self, key: str = None, category: str = None) -> str:
        if not key and not category:
            return "Error: Either 'key' or 'category' must be provided."
        
        count = 0
        if key:
            key_str = str(key)
            if key_str in self.worker.memories:
                del self.worker.memories[key_str]
                count += 1
                print(f"{self.C_CYAN}🧠 Memory Erased: {key}{self.C_RESET}")
        
        if category:
            cat_str = str(category)
            keys_to_delete = [k for k, v in self.worker.memories.items() 
                             if isinstance(v, dict) and v.get('category') == cat_str]
            for k in keys_to_delete:
                del self.worker.memories[k]
                count += 1
            if keys_to_delete:
                print(f"{self.C_CYAN}🧠 Memory Category Erased: {category} ({len(keys_to_delete)} items){self.C_RESET}")

        if count == 0:
            hint = ""
            if key:
                import difflib
                close = difflib.get_close_matches(str(key), list(self.worker.memories.keys()), n=3, cutoff=0.5)
                if close:
                    hint = f" Did you mean: {', '.join(close)}?"
                elif self.worker.memories:
                    hint = f" Stored keys: {', '.join(list(self.worker.memories.keys())[:10])}."
            return f"No memories found matching key='{key}' or category='{category}'.{hint}"
        return f"Successfully erased {count} memory item(s)."

class ListMemoriesTool(BaseTool):
    def __init__(self, worker):
        # Description will be loaded from prompt file by Worker.manager
        super().__init__(name="list_memories", description="Lists all stored memories, optionally filtered by category.")
        self.worker = worker

    def execute(self, category: str = None, scope: str = None) -> str:
        if not self.worker.memories:
            return "No memories recorded yet."
        
        results = []
        for k, v in self.worker.memories.items():
            if isinstance(v, dict):
                if category and v.get('category') != category:
                    continue
                if scope and v.get('scope', 'project') != scope:
                    continue
                if MemorizeTool.secret_error(k, v.get('value'), v.get('category')):
                    results.append(
                        f"[withheld] {k}: legacy secret-like memory is hidden; forget it and use a Nexus credential handle"
                    )
                    continue
                results.append(
                    f"[{v.get('scope', 'project')}/{v.get('category', 'general')}] "
                    f"{k}: {v.get('value')} (Saved: {v.get('timestamp')})"
                )
            else:
                if not category:
                    if MemorizeTool.secret_error(k, v, "legacy"):
                        results.append(f"[withheld] {k}: legacy secret-like memory is hidden")
                    else:
                        results.append(f"[legacy/project] {k}: {v}")
        
        if not results:
            return f"No memories found in category: {category}"
        
        return "\n".join(results)
