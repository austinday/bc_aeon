#!/usr/bin/env python3

import sys
sys.path.append('aeon')

from aeon.tools.flux_dev2 import FluxDev2Tool


tool = FluxDev2Tool()

print('Running FluxDev2Tool test...')
result = tool.generate_image(
    prompt='a photorealistic cat astronaut floating in space, Earth in the background, dramatic lighting',
    width=1024,
    height=1024,
    steps=28,
    guidance=3.5,
    shift=6.0,
    filename_prefix='test_flux_cat'
)
print(f'Generated image: {result}') 