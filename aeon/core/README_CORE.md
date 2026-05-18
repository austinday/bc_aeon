# Aeon Core Engine Guide

This directory contains the brain and execution loop of the Aeon agent (`worker.py` and `llm.py`).

## 1. Strict Prompt Separation
**Do not hardcode long prompt strings into Python files.**
All LLM-facing text, instructions, and tool descriptions must be stored as `.txt` files in `aeon/core/prompts/` and loaded centrally via `aeon/core/prompts/__init__.py`. 

## 2. JSON Block Notation (The `__BLOCK_N__` System)
Because LLMs struggle to output valid JSON when dealing with complex code snippets or heavily escaped strings, `llm.py` uses a custom Block substitution system. 
The LLM is instructed to output short placeholders (e.g., `__BLOCK_1__`) in the JSON, and append the actual raw text outside the JSON block. `llm.py` parses this and recombines it. 
If you modify `llm.py`, **do not break `_substitute_blocks` or `_extract_content_blocks`**.

## 3. The Execution Loop & Loop Detection
`worker.py` maintains a persistent Attempt Log. To prevent infinite loops, it tracks the hash of recent actions and their outputs. If an agent repeats the same action and gets the same output multiple times, a hard-coded intervention prompt is injected into the context to force the agent to use the `think` tool and pivot. Keep this in mind when tweaking the state manager.