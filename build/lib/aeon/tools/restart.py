import os
import json
from .base import BaseTool
from ..core.prompts import TOOL_DESC_RESTART_AEON

RESTART_STATE_PATH = '/tmp/aeon_restart_state.json'


class RestartAeonTool(BaseTool):
    """Saves agent state and signals a restart to apply code changes."""

    def __init__(self, worker):
        super().__init__(
            name='restart_aeon',
            description=TOOL_DESC_RESTART_AEON
        )
        self.worker = worker

    def execute(self, aeon_code_dir: str, reason: str = 'Applying code changes') -> str:
        if not aeon_code_dir:
            return 'Error: aeon_code_dir is required.'

        abs_dir = os.path.abspath(aeon_code_dir)
        if not os.path.isdir(abs_dir):
            return f'Error: Directory not found: {aeon_code_dir}'

        # Verify it looks like a Python package
        has_setup = os.path.exists(os.path.join(abs_dir, 'setup.py'))
        has_pyproject = os.path.exists(os.path.join(abs_dir, 'pyproject.toml'))
        if not has_setup and not has_pyproject:
            return (
                f'Error: {aeon_code_dir} does not appear to be a Python package '
                f'(no setup.py or pyproject.toml found).'
            )

        # Serialize worker state
        state = self.worker.serialize_state()
        state['aeon_code_dir'] = abs_dir
        state['original_cwd'] = os.getcwd()
        state['model_name'] = getattr(self.worker, 'model_name', None)
        state['debug_mode'] = getattr(self.worker, 'debug_mode', False)
        state['reason'] = reason

        try:
            with open(RESTART_STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            return f'Error saving restart state: {e}'

        return (
            f'Restart state saved. Reason: {reason}\n'
            f'The agent will now terminate and restart with updated code from {abs_dir}.\n'
            f'All memories and action history will be preserved.'
        )
