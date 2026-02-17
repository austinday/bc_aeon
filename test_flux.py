from aeon.tools.generate_image import GenerateImageTool

tool = GenerateImageTool()
result = tool.run(
    prompt="a photorealistic cat astronaut floating in space, Earth in the background, dramatic lighting",
    negative_prompt="",
    width=1024,
    height=1024,
    steps=20,
    guidance=4.0,
    shift=3.2,
    seed=-1,
    output_path="test_flux_cat.png"
)
print(result)
