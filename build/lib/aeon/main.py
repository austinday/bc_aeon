import os
import argparse
import json
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

def cli():
    """Command-line interface for the Aeon agent."""    
    try:
        # Parse arguments for restore
        parser = argparse.ArgumentParser(description='Aeon Agent CLI')
        parser.add_argument('--restore', type=str, help='Path to history file for restoration')
        parser.add_argument('--gemini', action='store_true', help='Use Google Gemini models instead of Grok')
        parser.add_argument('--gemini-flash', action='store_true', help='Use Google Gemini Flash for ALL models (fast/cheap)')
        args = parser.parse_args()

        # 1. Initialize Core Resources
        if args.gemini_flash:
            provider = "gemini-flash"
        elif args.gemini:
            provider = "gemini"
        else:
            provider = "grok"
            
        llm_client = LLMClient(provider=provider)
        
        # 2. Initialize Worker WITHOUT tools first
        # This allows us to pass the 'worker' instance to tools that need it (dependency injection)
        worker = Worker(llm_client=llm_client)
        worker.max_history_tokens = max(int(os.getenv("AEON_MAX_TOKENS", "30000")), 25000)

        # 3. Define the Dependency Map
        # These are the objects available to any tool's __init__ method
        deps = {
            'llm_client': llm_client,
            'worker': worker,
            'max_tokens': worker.max_history_tokens 
        }

        # 4. Dynamically Load Tools
        print("Loading tools...")
        tools = load_tools_from_directory(package_name="aeon.tools", dependencies=deps)
        
        # 5. Register loaded tools with the worker
        worker.register_tools(tools)

        print(f"Aeon Agent is ready with {len(worker.tools)} tools loaded (Provider: {provider}).")
        print("Type your objective. Type 'exit' or CTRL+d to quit. Use /clear to reset the agent and start fresh.")
        while True:
            try:
                objective = input("> ")
                if objective.strip() == '/clear':
                    print("Resetting agent...")
                    llm_client = LLMClient(provider=provider)
                    worker = Worker(llm_client=llm_client)
                    worker.max_history_tokens = max(int(os.getenv("AEON_MAX_TOKENS", "30000")), 25000)
                    
                    deps['llm_client'] = llm_client
                    deps['worker'] = worker
                    deps['max_tokens'] = worker.max_history_tokens
                    
                    tools = load_tools_from_directory("aeon.tools", deps)
                    worker.register_tools(tools)
                    
                    print('Agent reset. Ready for a new objective.')
                    continue
                if objective.lower().strip() in ['exit', 'quit']:
                    print("Goodbye!")
                    break

                worker.run(objective=objective)
            except KeyboardInterrupt:
                print("Interrupted. Returning to the prompt. Press CTRL+d to terminate the session.")
                continue
            except EOFError:
                break
                
    except (json.JSONDecodeError, IOError, ValueError) as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    cli()