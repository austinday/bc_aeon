from PIL import Image
from pathlib import Path

IMAGE_PATH = Path("aeon_output/browser_validation/navigate_clean.jpg")

def main():
    if not IMAGE_PATH.exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    with Image.open(IMAGE_PATH) as img:
        width, height = img.size
        print(f"Image: {IMAGE_PATH}")
        print(f"Dimensions: {width}x{height}")
        
        if width == 3840 and height == 2160:
            print("SUCCESS: Resolution is doubled (3840x2160)!")
        elif width == 1920 and height == 1080:
            print("FAILURE: Resolution is still 1920x1080.")
        else:
            print(f"UNEXPECTED: Resolution is {width}x{height}.")

if __name__ == "__main__":
    main()