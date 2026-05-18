import json
import inspect
import comfy.nodes

def inspect_nodes():
    all_nodes = comfy.nodes.NODE_CLASS_MAPPINGS
    results = []
    
    for node_name, node_class in all_nodes.items():
        # We are looking for nodes that output CONDITIONING
        # and are likely related to LTX or T5 encoding.
        
        # Get return types
        return_types = getattr(node_class, 'RETURN_TYPES', ())
        if isinstance(return_types, str):
            return_types = (return_types,)
            
        if 'CONDITIONING' in return_types:
            # Filter for LTX or T5 related nodes to reduce noise
            if any(keyword in node_name.upper() for keyword in ['LTX', 'T5', 'ENCODE', 'TEXT']):
                
                # Get input types
                input_types = {}
                if hasattr(node_class, 'INPUT_TYPES'):
                    try:
                        # INPUT_TYPES is often a function
                        if callable(node_class.INPUT_TYPES):
                            input_types = node_class.INPUT_TYPES()
                        else:
                            input_types = node_class.INPUT_TYPES
                    except Exception as e:
                        input_types = {"error": str(e)}
                
                # Extract required inputs
                required = {}
                if isinstance(input_types, dict) and 'required' in input_types:
                    required = input_types['required']
                
                results.append({
                    "node_name": node_name,
                    "class": node_class.__name__,
                    "inputs": required,
                    "outputs": return_types
                })
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    inspect_nodes()