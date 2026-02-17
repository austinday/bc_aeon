#!/usr/bin/env python3

import sys
sys.path.append('aeon')

from aeon.tools.generate_image import GenerateImageTool

tool = GenerateImageTool()

prompt = 'a photorealistic cat astronaut floating in space, Earth in the background, dramatic lighting, high quality'

result = tool.execute(
    prompt=prompt,
    width=1024,
    height=1024,
    steps=50,
    cfg_scale=1.0,
    seed=-1,
    output_path='test_flux2.png'
)

print(result)