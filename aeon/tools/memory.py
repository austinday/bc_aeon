from datetime import datetime
from .base import BaseTool
from ..core.prompts import TOOL_DESC_MEMORIZE, TOOL_DESC_FORGET

class MemorizeTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name="memorize", description=TOOL_DESC_MEMORIZE)
        self.worker = worker

    def execute(self, key: str, value: str, category: str = "general") -> str:
        if not key or not value:
            return "Error: Both 'key' and 'value' are required."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.worker.memories[str(key)] = {
            "value": str(value),
            "category": str(category),
            "timestamp": timestamp
        }
        print(f"{self.C_CYAN}🧠 Memory Saved: [{category}] {key} = {value}{self.C_RESET}")
        return f"Memorized: [{category}] {key} = {value}"

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
            return f"No memories found matching key='{key}' or category='{category}'."
        return f"Successfully erased {count} memory item(s)."

class ListMemoriesTool(BaseTool):
    def __init__(self, worker):
        # Description will be loaded from prompt file by Worker.manager
        super().__init__(name="list_memories", description="Lists all stored memories, optionally filtered by category.")
        self.worker = worker

    def execute(self, category: str = None) -> str:
        if not self.worker.memories:
            return "No memories recorded yet."
        
        results = []
        for k, v in self.worker.memories.items():
            if isinstance(v, dict):
                if category and v.get('category') != category:
                    continue
                results.append(f"[{v.get('category', 'general')}] {k}: {v.get('value')} (Saved: {v.get('timestamp')})")
            else:
                if not category:
                    results.append(f"{k}: {v}")
        
        if not results:
            return f"No memories found in category: {category}"
        
        return "\n".join(results)