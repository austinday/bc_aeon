# Aeon Agent - Enhanced Memory System

The Aeon agent now features a structured persistent memory system that allows it to store, retrieve, and prune key information across iterations.

## Features
- **Structured Storage**: Memories are stored as objects containing a value, a category, and a timestamp.
- **Categorization**: The agent can organize memories into categories (e.g., `planning`, `credentials`, `summaries`) to better manage context.
- **Targeted Pruning**: The `forget` tool can remove specific memories by key or wipe entire categories to prevent context rot.
- **Introspection**: The `list_memories` tool allows the agent to review its current knowledge base, optionally filtered by category.

## Tools
- `memorize(key, value, category="general")`: Saves a piece of information.
- `forget(key=None, category=None)`: Erases memories by key or category.
- `list_memories(category=None)`: Lists stored memories.

## Implementation Details
- **Worker Integration**: `aeon/core/worker.py` has been updated to format structured memories into a human-readable list in the agent's prompt.
- **Tool Logic**: Implemented in `aeon/tools/memory.py`.