import re
import json

def analyze_nodes(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by the node delimiter
    nodes_raw = content.split('=== NODE: ')
    
    results = []
    for node_raw in nodes_raw[1:]:
        lines = node_raw.split('\n')
        node_name = lines[0].strip(' =')
        
        # Try to find the JSON part
        json_str = '\n'.join(lines[1:])
        try:
            # The file might have trailing text or be slightly malformed JSON
            # We'll try to find the first '{' and last '}'
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != -1:
                node_data = json.loads(json_str[start:end])
                
                outputs = node_data.get('output', {})
                inputs = node_data.get('input', {})
                
                # Check if any output is CONDITIONING
                has_conditioning_output = any('CONDITIONING' in str(val) for val in outputs.values())
                
                if has_conditioning_output:
                    # Check if it takes STRING as input
                    takes_string = False
                    # Check required inputs
                    req = inputs.get('required', {})
                    for in_name, in_val in req.items():
                        if 'STRING' in str(in_val):
                            takes_string = True
                            break
                    
                    # Check optional inputs
                    opt = inputs.get('optional', {})
                    for in_name, in_val in opt.items():
                        if 'STRING' in str(in_val):
                            takes_string = True
                            break
                            
                    results.append({
                        "node": node_name,
                        "takes_string": takes_string,
                        "outputs": outputs,
                        "inputs": inputs
                    })
        except Exception as e:
            continue

    return results

if __name__ == "__main__":
    file_path = 'scripts/debug/ltxv_relevant_nodes.txt'
    found = analyze_nodes(file_path)
    
    print(f"Found {len(found)} nodes that output CONDITIONING.\n")
    for item in found:
        status = "[MATCH]" if item['takes_string'] else "[NO STRING]"
        print(f"{status} Node: {item['node']}")
        if item['takes_string']:
            print(f"  Inputs: {json.dumps(item['inputs'], indent=2)}")
            print(f"  Outputs: {json.dumps(item['outputs'], indent=2)}")
            print("-" * 40)