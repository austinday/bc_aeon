import json
import sys

def extract_node(node_name):
    file_path = 'scripts/debug/ltxv_nodes.txt'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if node_name in data:
            print(json.dumps(data[node_name], indent=2))
        else:
            print(f"Node {node_name} not found in {file_path}")
            # List all nodes to help debugging
            print("\nAvailable nodes:")
            for name in data.keys():
                print(name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_node_info.py <node_name>")
    else:
        extract_node(sys.argv[1])