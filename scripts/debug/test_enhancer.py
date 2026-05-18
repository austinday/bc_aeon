import sys
import os

# Add current directory to path to allow importing aeon
sys.path.append(os.getcwd())

from aeon.core.utils.prompt_enhancer import AdvancedPromptEnhancer, PromptEnhancer

class MockLLM:
    def generate(self, system_prompt, user_prompt):
        print(f"[MockLLM] System: {system_prompt[:50]}...")
        print(f"[MockLLM] User: {user_prompt}")
        return "A hyper-realistic, cinematic masterpiece of a futuristic city with neon lights, rainy streets, and deep reflections, 8k resolution, highly detailed."

def test_advanced_enhancer():
    print("Testing AdvancedPromptEnhancer with Mock LLM...")
    mock_llm = MockLLM()
    enhancer = AdvancedPromptEnhancer(llm_instance=mock_llm)
    
    test_prompt = "a futuristic city"
    result = enhancer.enhance(test_prompt)
    
    print(f"Original: {test_prompt}")
    print(f"Enhanced: {result}")
    
    if "hyper-realistic" in result and "cinematic" in result:
        print("SUCCESS: LLM expansion worked.")
    else:
        print("FAILURE: LLM expansion did not produce expected mock output.")
        sys.exit(1)

def test_fallback():
    print("\nTesting Fallback to Basic Enhancer...")
    # Pass None and ensure it doesn't crash, but falls back to PromptEnhancer
    enhancer = AdvancedPromptEnhancer(llm_instance=None)
    
    # We simulate the import error or LLM failure by not providing a valid client
    # The current implementation of AdvancedPromptEnhancer will try to import LLMClient and fail
    # which should trigger the except block and call PromptEnhancer().enhance()
    result = enhancer.enhance("a cat")
    print(f"Original: a cat")
    print(f"Enhanced (Fallback): {result}")
    
    if "image of a cat" in result or "highly detailed" in result:
        print("SUCCESS: Fallback mechanism worked.")
    else:
        print("FAILURE: Fallback mechanism failed.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_advanced_enhancer()
        test_fallback()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)