import re

file_path = "/home/aday/bc_aeon/aeon_output/debug/ltxv_nodes_source.py"

with open(file_path, "r") as f:
    content = f.read()

# Search for classes that look like loaders
loader_pattern = re.compile(r"class\s+(\w+)\(.*?):")
matches = loader_pattern.findall(content)

print("Found classes in ltxv_nodes_source.py:")
for match in matches:
    print(match)

# Search for specific keywords related to text encoding and connectors
keywords = ["CLIP", "T5", "Encoder", "Connector", "Embeddings", "Load"]
for kw in keywords:
    print(f"\n--- Searching for {kw} ---")
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if kw.lower() in line.lower():
            # Print the line and a few lines around it for context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                prefix = ">> " if j == i else "   "
                print(f"{prefix}{lines[j]}")
            print("-" * 20)

# Specifically look for how the connector is loaded
print("\n--- Searching for connector loading logic ---")
connector_matches = re.findall(r"class\s+(\w+).*?load.*?connector", content, re.DOTALL | re.IGNORECASE)
if connector_matches:
    print("Potential connector loaders:", connector_matches)
else:
    # Try a broader search for 'connector'
    connector_indices = [i for i, line in enumerate(lines) if "connector" in line.lower()]
    for idx in connector_indices:
        print(f"Line {idx}: {lines[idx]}")
