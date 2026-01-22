"""
Abstract base class for all tools in the Aeon agent.
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    # ANSI Color Codes for standardized output across tools
    C_RED = '\033[91m'
    C_YELLOW = '\033[93m'
    C_CYAN = '\033[96m'
    C_GREEN = '\033[92m'
    C_BLUE = '\033[94m'
    C_RESET = '\033[0m'

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        # pylint: disable=unnecessary-pass
        pass

    def format_error_message(
        self,
        error: Exception,
        context: str,
        resolution: str = 'retrying with adjusted parameters'
    ) -> str:
        """Format error into a yellow-colored explanatory message."""
        reason = str(error).splitlines()[0] if str(error) else 'Unknown'
        return (
            f'{self.C_YELLOW}ERROR: Encountered {type(error).__name__} while {context}. '
            f'Reason: {reason}. Resolving by {resolution}.{self.C_RESET}'
        )
