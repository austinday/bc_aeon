"""
Abstract base class for all tools in the Aeon agent.
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all tools."""
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
        c_yellow = '\033[93m'
        c_reset = '\033[0m'
        reason = str(error).splitlines()[0] if str(error) else 'Unknown'
        return (
            f'{c_yellow}ERROR: Encountered {type(error).__name__} while {context}. '
            f'Reason: {reason}. Resolving by {resolution}.{c_reset}'
        )

