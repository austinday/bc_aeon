from huggingface_hub import HfApi
from pathlib import Path
import sys

def check():
    token_path = Path.home() / 'huggingface_access_token.txt'
    if not token_path.exists():
        print("FAIL: No token file found.")
        return
    
    with open(token_path, 'r') as f:
        token = f.read().strip()
        
    api = HfApi(token=token)
    
    print(f"Checking token identity...")
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print(f"Token invalid: {e}")
        return

    print("\nChecking Flux.1-Schnell access...")
    try:
        api.model_info("black-forest-labs/FLUX.1-schnell")
        print("SUCCESS: You have full access.")
    except Exception as e:
        err = str(e)
        if "403" in err or "401" in err or "404" in err:
            print("\n!!! ACCESS DENIED !!!")
            print("You must accept the license agreement here:")
            print("https://huggingface.co/black-forest-labs/FLUX.1-schnell")
        else:
            print(f"Error: {err}")

if __name__ == "__main__":
    check()