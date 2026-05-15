import json

with open('scripts/debug/ltx_nodes_details.json', 'r') as f:
    data = json.load(f)

print(f"{'Node Name':<30} | {'Inputs':<60} | {'Outputs'}")
print("-" * 110)

for node_name, details in data.items():
    if 'LTX' in node_name.upper():
        inputs = details.get('input', {}).get('required', {})
        outputs = details.get('output', [])
        
        # Simplify inputs for display
        input_summary = ", ".join([f"{k}({v[0]})" for k, v in inputs.items()])
        output_summary = ", ".join(outputs)
        
        print(f"{node_name:<30} | {input_summary:<60} | {output_summary}")