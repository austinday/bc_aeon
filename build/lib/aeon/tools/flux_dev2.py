from aeon.comfyui.backend import ComfyUIBackend

class FluxDev2Tool:
    def __init__(self):
        self.backend = ComfyUIBackend()

    def generate_image(self, prompt, width=1024, height=1024, steps=28, guidance=4.0, shift=6.0, seed=-1, filename_prefix='flux_dev2'):
        params = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'guidance': guidance,
            'shift': shift,
            'seed': seed,
            'filename_prefix': filename_prefix
        }
        return self.backend.run_model('flux_image', params)
