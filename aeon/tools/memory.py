from .base import BaseTool
from ..core.prompts import (
    TOOL_DESC_MEMORIZE_DETAIL,
    TOOL_DESC_FORGET_DETAIL
)

class MemorizeDetailTool(BaseTool):
    """A tool to memorize a detail into the persistent context."""
    def __init__(self, worker):
        super().__init__(
            name="memorize_detail",
            description=TOOL_DESC_MEMORIZE_DETAIL
        )
        self.worker = worker

    def execute(self, key: str, value: str) -> str:
        if not key or not value:
            return "Error: Both 'key' and 'value' parameters are required."
        self.worker.memories[key] = value
        return f"Successfully memorized detail '{key}'."

class ForgetDetailTool(BaseTool):
    """A tool to forget a previously memorized detail."""
    def __init__(self, worker):
        super().__init__(
            name="forget_detail",
            description=TOOL_DESC_FORGET_DETAIL
        )
        self.worker = worker

    def execute(self, key: str) -> str:
        if not key:
            return "Error: 'key' parameter is required."
        if key in self.worker.memories:
            del self.worker.memories[key]
            return f"Successfully forgot detail '{key}'."
        return f"Error: Memory key '{key}' not found in memories."
