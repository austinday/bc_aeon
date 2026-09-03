import os
import json
from pathlib import Path
from .base import BaseTool
from ..core.prompts import TOOL_DESC_RESTART_AEON

RESTART_STATE_PATH = f'/tmp/aeon_restart_state_{os.getpid()}.json'


def restart_validation_boundary_available() -> bool:
    """Loaded-code latch for the not-yet-proven candidate import sandbox."""

    return False


class RestartAeonTool(BaseTool):
    """Saves agent state and signals a restart to apply code changes."""

    def __init__(self, worker):
        super().__init__(
            name='restart_aeon',
            description=TOOL_DESC_RESTART_AEON
        )
        self.worker = worker

    def _default_code_dir(self) -> str:
        """The aeon source root, derived from the installed package location.

        The agent should not have to know or supply the path to its own source;
        aeon.core.paths already resolves the install/source root (the dir holding
        setup.py) independent of the current workspace. Used when the caller omits
        aeon_code_dir so a restart 'just works' from any workspace."""
        try:
            from ..core.paths import PROJECT_ROOT
            return str(PROJECT_ROOT)
        except Exception:
            return ''

    def execute(self, aeon_code_dir: str = None, reason: str = 'Applying code changes') -> str:
        # aeon_code_dir is optional: when omitted, auto-derive the source root
        # from the installed package so the model never has to hand-supply (and
        # potentially mis-type) the path to its own code.
        if not aeon_code_dir:
            aeon_code_dir = self._default_code_dir()
            if not aeon_code_dir:
                return ('Error: aeon_code_dir was not provided and the source root could not be '
                        'auto-derived. Pass the absolute path to the Aeon source tree (the directory '
                        'containing setup.py).')

        canonical = self._default_code_dir()
        if not canonical:
            return 'Error: canonical Aeon source root could not be resolved.'
        try:
            abs_dir = str(Path(aeon_code_dir).expanduser().resolve(strict=True))
            canonical_dir = str(Path(canonical).expanduser().resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            return f'Error: Invalid Aeon source directory: {exc}'
        if abs_dir != canonical_dir:
            return (
                f'Error: restart_aeon only reloads its canonical source tree '
                f'({canonical_dir}); alternate package installation is not allowed.'
            )

        # Verify it looks like a Python package
        has_setup = os.path.exists(os.path.join(abs_dir, 'setup.py'))
        has_pyproject = os.path.exists(os.path.join(abs_dir, 'pyproject.toml'))
        if not has_setup and not has_pyproject:
            return (
                f'Error: {aeon_code_dir} does not appear to be a Python package '
                f'(no setup.py or pyproject.toml found).'
            )

        if not restart_validation_boundary_available():
            return (
                "Error: restart is blocked because this host does not yet have "
                "an actively-probed masked-home sandbox for importing modified "
                "Aeon source. No restart state was written and no candidate code "
                "was executed."
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
            f'The agent will now terminate and reload updated canonical source from {abs_dir}.\n'
            f'All memories and action history will be preserved.'
        )
