import re
import os
import json
from typing import Optional, Dict, Any, Union
from openai import OpenAI
from .ollama_client import OllamaClient

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    PYDANTIC_AVAILABLE = False

class DictAsObject:
    """Wrapper to make dict behave like an object with attribute access"""
    def __init__(self, data: dict):
        self.__dict__.update(data)
        self._data = data
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class PydanticHandler:
    """Handle Pydantic model related logic"""
    
    @staticmethod
    def generate_schema_instruction(pydantic_model):
        """Generate JSON schema instruction for structured output"""
        if not PYDANTIC_AVAILABLE or not hasattr(pydantic_model, 'model_json_schema'):
            return "Please respond with valid JSON format."
        
        try:
            schema = pydantic_model.model_json_schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            instruction = "Respond with JSON format:\n{"
            for field_name, field_info in properties.items():
                field_type = field_info.get('type', 'any')
                is_required = field_name in required
                req_text = "required" if is_required else "optional"
                instruction += f'\n  "{field_name}": {field_type} ({req_text})'
            instruction += "\n}"
            return instruction
        except Exception:
            return "Please respond with valid JSON format."
    
    @staticmethod
    def parse_to_model(data: dict, pydantic_model):
        """Try to parse dictionary to Pydantic model"""
        if not PYDANTIC_AVAILABLE:
            return DictAsObject(data)
        
        try:
            return pydantic_model.model_validate(data)
        except Exception:
            try:
                return pydantic_model(**data)
            except Exception:
                return DictAsObject(data)

class LLMClient:
    def __init__(self, args=None):
        """
        Initialize the LLM client.
        
        Args:
            args (optional): Configuration arguments containing:
                - llm_provider: "openai", "ollama", or "openrouter"
                - model_name: Name of the model to use
                - api_key: API key for OpenAI/OpenRouter (if using OpenAI/OpenRouter)
                - base_url: Base URL for Ollama (if using Ollama)
        """
        self.provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        
        # Set default models based on provider
        if self.provider == 'openai':
            default_model = 'gpt-4o-mini'
        elif self.provider == 'openrouter':
            default_model = 'openai/gpt-4o-mini'  # OpenRouter format: provider/model
        else:
            default_model = 'llama2'
            
        self.model = os.getenv('LLM_MODEL', default_model)
        
        if self.provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', None))
        elif self.provider == 'openrouter':
            self.client = OpenAI(
                api_key=os.getenv('OPENROUTER_API_KEY', None),
                base_url="https://openrouter.ai/api/v1"
            )
        elif self.provider == 'ollama':
            self.client = OllamaClient(model_name=self.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _ensure_json_in_messages(self, messages):
        """Ensure messages contain 'json' keyword to satisfy OpenAI JSON mode requirement"""
        json_mentioned = any('json' in msg.get('content', '').lower() for msg in messages)
        if not json_mentioned:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += " Please respond in JSON format."
            else:
                messages.insert(0, {"role": "system", "content": "Please respond in JSON format."})
        return messages

    def chat_completion(self, 
                       prompt: str = None,
                       system_prompt: str = "You are a helpful assistant.", 
                       messages: list = None,
                       response_format = None,
                       json_response: bool = False, 
                       model: Optional[str] = None,
                       temperature: float = 0.0) -> Union[str, Dict, Any]:
        """
        Send a chat completion request to the LLM.
        
        Args:
            prompt (str, optional): The user prompt
            system_prompt (str): The system prompt  
            messages (list, optional): List of messages in OpenAI format
            response_format (optional): Pydantic model for structured output
            json_response (bool): Whether to expect JSON response
            model (str, optional): Override the default model
            temperature (float): Temperature for response generation
            
        Returns:
            Union[str, Dict, Any]: The model's response
        """
        # Build messages
        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        # Handle structured output
        if response_format is not None:
            schema_instruction = PydanticHandler.generate_schema_instruction(response_format)
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{schema_instruction}"
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            json_response = True
        
        # Get response
        if self.provider in ['openai', 'openrouter']:
            if json_response:
                messages = self._ensure_json_in_messages(messages)
            
            response = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"} if json_response else None
            )
            content = response.choices[0].message.content
            
        elif self.provider == 'ollama':
            # Extract prompts for Ollama
            user_prompt = ""
            system_content = "You are a helpful assistant."
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_prompt += msg["content"]
            
            content = self.client.chat_completion(
                prompt=user_prompt,
                system_prompt=system_content,
                json_response=json_response,
                temperature=temperature,
                model=model or self.model
            )
        
        # Parse response
        if json_response or response_format is not None:
            try:
                parsed_json = json.loads(content) if isinstance(content, str) else content
                
                if response_format is not None:
                    return PydanticHandler.parse_to_model(parsed_json, response_format)
                return parsed_json
                
            except json.JSONDecodeError:
                # Try to extract JSON from code blocks
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(1))
                        if response_format is not None:
                            return PydanticHandler.parse_to_model(parsed_json, response_format)
                        return parsed_json
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f"Failed to parse JSON response: {content}")
        
        return content
