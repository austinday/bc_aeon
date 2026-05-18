from aeon.tools.vision import AnalyzeImageTool
import os

def solve():
    # The path is derived from the last browser output
    image_path = "/home/aday/.aeon/temp/browser_output_2151344_popup_11/overlay.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    tool = AnalyzeImageTool()
    prompt = (
        "This is a Google reCAPTCHA image challenge. "
        "Look at the image and identify which of the numbered red boxes (from [2] to [10]) "
        "contain the objects requested by the challenge (e.g., traffic lights, crosswalks, buses). "
        "List only the numbers of the boxes that should be clicked. "
        "If you are unsure, provide your best guess."
    )
    
    print(f"Analyzing image: {image_path}...")
    result = tool.execute(image_path=image_path, prompt=prompt)
    print("\n--- VISION RESULT ---")
    print(result)
    print("--- END RESULT ---")

if __name__ == "__main__":
    solve()