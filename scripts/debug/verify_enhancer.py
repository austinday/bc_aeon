import sys
import os

# Add current directory to path to ensure we import the local version of aeon
sys.path.append(os.getcwd())

try:
    from aeon.core.utils.prompt_enhancer import AdvancedPromptEnhancer, PromptEnhancer
    print("Successfully imported enhancers.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class MockLLM:
    def __init__(self):
        self.primary_client = self.MockClient()
        self.primary_model = "mock-model"

    class MockClient:
        def chat(self):
            return self
        def completions(self):
            return self
        def create(self, model, messages, temperature=0.7, response_format=None, stream=False):
            class MockResponse:
                def __init__(self):
                    self.choices = [
                        type('Choice', (), {
                            'message': type('Msg', (), {'content': 'A highly detailed, cinematic, 8k masterpiece of a futuristic city with neon lights and rain-slicked streets, volumetric lighting, hyper-realistic textures.'})()
                        })
                    ]
            return MockResponse()

def test_enhancer():
    print("\n--- Testing AdvancedPromptEnhancer with Mock LLM ---")
    mock_llm = MockLLM()
    enhancer = AdvancedPromptEnhancer(llm_instance=mock_llm)
    
    prompt = "a futuristic city"
    enhanced = enhancer.enhance(prompt)
    print(f"Original: {prompt}")
    print(f"Enhanced: {enhanced}")
    
    if "cinematic" in enhanced and "futuristic city" in enhanced:
        print("SUCCESS: LLM expansion worked.")
    else:
        print("FAILURE: LLM expansion failed or returned unexpected result.")

    print("\n--- Testing Fallback to Basic Enhancer ---")
    # Pass None to trigger the try-except block in enhance() because LLMClient(None, None) will fail
    fallback_enhancer = AdvancedPromptEnhancer(llm_instance=None)
    # We simulate a failure by not having a valid LLMClient configured in the environment
    # The code should catch the exception and use PromptEnhancer()
    prompt_simple = "a cat"
    enhanced_fallback = fallback_enhancer.enhance(prompt_simple)
    print(f"Original: {prompt_simple}")
    print(f"Enhanced (Fallback): {enhanced_fallback}")
    
    if len(enhanced_fallback) > len(prompt_simple):
        print("SUCCESS: Fallback mechanism worked.")
    else:
        print("FAILURE: Fallback mechanism failed.")

if __name__ == "__main__":
    test_enhancer()