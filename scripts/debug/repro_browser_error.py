import base64
import os
from aeon.tools.browser import process_browser_response

def test_repro():
    print("Testing process_browser_response for NameError...")
    # Create a dummy image for the base64 data
    from PIL import Image
    import io
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    dummy_data = {
        "status": "success",
        "clean_b64": base64.b64encode(img_bytes).decode(),
        "overlay_b64": base64.b64encode(img_bytes).decode(),
        "elements": [],
        "markdown": "Test Markdown Content"
    }
    
    try:
        # We use a dummy session_id and tab_id
        result = process_browser_response(dummy_data, "Test Action", "12345", "test_tab")
        print("SUCCESS: process_browser_response executed without NameError.")
        print(result)
    except NameError as e:
        print(f"FAILED: Caught expected NameError: {e}")
        raise e
    except Exception as e:
        print(f"FAILED: Caught unexpected exception: {type(e).__name__}: {e}")
        raise e

if __name__ == "__main__":
    test_repro()