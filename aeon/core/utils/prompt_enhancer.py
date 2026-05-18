import random

class PromptEnhancer:
    """
    Utility to enhance simple prompts for FLUX.1 models using best practices:
    - Natural language descriptions.
    - Quality modifiers.
    - Lighting and composition details.
    """
    def __init__(self):
        self.quality_boosters = [
            "highly detailed", "masterpiece", "8k resolution", "photorealistic", 
            "extremely la realistic", "extremely intricate", "sharp focus", "professional photography", 
            "cinematic lighting", "hyper-realistic", "ultra-detailed"
        ]
        
        self.styles = [
            "soft cinematic lighting", "depth of field", "bokeh background", 
            "global illumination", "ray-traced reflections", "intricate textures"
        ]

    def enhance(self, prompt: str) -> str:
        if not prompt or len(prompt.strip()) == 0:
            return prompt

        # If the prompt is already very long, it might be a detailed prompt already
        if len(prompt) > 200:
            return prompt

        # Build the enhanced prompt
        # FLUX prefers natural language, so we blend the boosters into a descriptive sentence
        boosters = random.sample(self.quality_boosters, k=random.randint(2, 4))
        style_mods = random.sample(self.styles, k=random.randint(1, 2))
        
        enhancement_prefix = f"A {', '.join(boosters)} image of "
        enhancement_suffix = f". {', '.join(style_mods)}."
        
        enhanced = f"{enhancement_prefix}{prompt}{enhancement_suffix}"
        
        return enhanced