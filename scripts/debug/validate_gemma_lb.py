import sys
sys.path.insert(0, "/home/aday/bc_aeon/aeon/scripts")
import gemma_lb

print("Import successful")
print("httpx limits:", gemma_lb.client._limits)
print("http2 enabled:", gemma_lb.client._transport._pool._http2)
print("NODE0_URL:", gemma_lb.NODE0_URL)
print("NODE1_URL:", gemma_lb.NODE1_URL)
print("CONTEXT_LIMIT_NODE1:", gemma_lb.CONTEXT_LIMIT_NODE1)
print("Sanity check passed.")