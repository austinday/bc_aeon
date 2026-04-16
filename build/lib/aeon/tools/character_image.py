import os
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_SCENE_WITH_CHARACTERS

class GenerateSceneWithCharactersTool(BaseTool):
    '''A tool to generate consistent characters using FLUX.2 native multi-reference.'''
    def __init__(self):
        super().__init__(
            name='generate_scene_with_characters',
            description=TOOL_DESC_GENERATE_SCENE_WITH_CHARACTERS
        )

    def execute(self, prompt: str, characters: list, output_path: str, width: int = 1024, height: int = 1024) -> str:
        return 'ERROR: The character_image tool is currently disabled pending the FLUX.2 native multi-reference workflow update. Please provide the workflow_api.json from ComfyUI so I can wire up the native FLUX.2 image conditioning nodes.'
