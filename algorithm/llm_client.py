from openai import OpenAI

class LLMClient:
    def __init__(self, args):
        self.client = OpenAI(
            organization=args.organization, 
            project=args.project, 
            api_key=args.apikey
        )
        self.model = "gpt-4o-mini"  # Default model

    def chat_completion(self, prompt, system_prompt="You are a helpful assistant.", json_response=False):
        """
        Send a chat completion request to the LLM.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str): The system prompt
            json_response (bool): Whether to expect JSON response
        
        Returns:
            str: The model's response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
