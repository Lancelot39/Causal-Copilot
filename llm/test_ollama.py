import os
import sys
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.llm]

if os.getenv("RUN_OLLAMA_TESTS") != "1":
    pytest.skip("set RUN_OLLAMA_TESTS=1 to run Ollama integration checks", allow_module_level=True)

from llm.ollama_client import OllamaClient

def test_ollama_client():
    try:
        # Initialize the client
        print("Initializing Ollama client...")
        client = OllamaClient(model_name="llama2")
        
        # List available models first
        print("\nChecking available models...")
        models = client.list_models()
        print("Available models:", models)
        
        # Test basic chat completion
        print("\nTesting basic chat completion...")
        response = client.chat_completion(
            prompt="What is causal inference?",
            system_prompt="You are an expert in causal inference and statistics."
        )
        print("Basic chat response:", response)
        
        # Test JSON response with more specific prompt
        print("\nTesting JSON response...")
        json_response = client.chat_completion(
            prompt="Create a JSON object with exactly three key concepts in causal inference. Each concept should have a 'name' and 'description' field.",
            json_response=True
        )
        print("JSON response:", json_response)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    test_ollama_client()
