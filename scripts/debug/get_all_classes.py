import requests
import json

def get_classes():
    try:
        response = requests.get("http://localhost:8188/object/class")
        if response.status_code == 200:
            classes = response.json()
            with open("scripts/debug/live_classes.json", "w") as f:
                json.dump(classes, f, indent=4)
            print(f"Successfully retrieved {len(classes)} classes. Saved to scripts/debug/live_classes.json")
            
            # Print LTX related classes to console for immediate feedback
            ltx_classes = [c for c in classes if 'LTX' in c.upper()]
            latent_classes = [c for c in classes if 'LATENT' in c.upper()]
            print("\nLTX related classes:")
            for c in ltx_classes: print(c)
            print("\nLatent related classes:")
            for c in latent_classes: print(c)
        else:
            print(f"Failed to retrieve classes: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_classes()