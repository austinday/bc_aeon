import sys
from aeon.main import CLOUD_MODELS

def test_grok_update():
    grok_models = [m['model'] for m in CLOUD_MODELS if m['provider'] == 'grok']
    print(f"Found Grok models: {grok_models}")
    
    if 'grok-4.3-latest' in grok_models:
        print("SUCCESS: grok-4.3-latest is present.")
        sys.exit(0)
    else:
        print("FAILURE: grok-4.3-latest not found in CLOUD_MODELS.")
        sys.exit(1)

if __name__ == "__main__":
    test_grok_update()