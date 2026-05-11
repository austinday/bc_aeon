import os
from PIL import Image
from pathlib import Path

def check_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image: {image_path} | Resolution: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

def main():
    output_dir = Path("aeon_output/browser_validation")
    images = list(output_dir.glob("*.jpg"))
    
    if not images:
        print("No screenshots found in aeon_output/browser_validation/")
        return

    all_correct = True
    for img_path in images:
        res = check_resolution(img_path)
        if res != (3840, 2160):
            all_correct = False
    
    if all_correct:
        print("\nSUCCESS: All screenshots have the expected resolution of 3840x2160.")
    else:
        print("\nFAILURE: Some screenshots do not have the expected resolution of 3840x2160.")

if __name__ == "__main__":
    main()