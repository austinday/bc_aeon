from .base import BaseTool
from ..core.prompts import TOOL_DESC_MEMORIZE, TOOL_DESC_FORGET

class MemorizeTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name="memorize", description=TOOL_DESC_MEMORIZE)
        self.worker = worker

    def execute(self, key: str, value: str) -> str:
        if not key or not value:
            return "Error: Both 'key' and 'value' are required."
        self.worker.memories[str(key)] = str(value)
        print(f"{self.C_CYAN}🧠 Memory Saved: {key} = {value}{self.C_RESET}")
        return f"Memorized: {key} = {value}"

class ForgetTool(BaseTool):
    def __init__(self, worker):
        super().__init__(name="forget", description=TOOL_DESC_FORGET)
        self.worker = worker

    def execute(self, key: str) -> str:
        if not key:
            return "Error: 'key' is required."
        key_str = str(key)
        if key_str in self.worker.memories:
            del self.worker.memories[key_str]
            print(f"{self.C_CYAN}🧠 Memory Erased: {key}{self.C_RESET}")
            return f"Forgot memory for key: {key}"
        else:
            return f"Memory key '{key}' not found."
