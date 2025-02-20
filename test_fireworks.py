import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional
from datetime import datetime

class FireworksModel:
    def __init__(self, model_id: str, max_tokens: int = 16384, temperature: float = 0.6):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.responses: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": self.temperature,
        }

class FireworksAPI:
    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    def __init__(self):
        self.api_key = self._load_api_key()
        self.models = {
            "deepseek": FireworksModel("accounts/fireworks/models/deepseek-r1"),
            "llama": FireworksModel("accounts/fireworks/models/llama-v3p1-8b-instruct"),
            "deepseek-v3": FireworksModel("accounts/fireworks/models/deepseek-v3")
        }

    def _load_api_key(self) -> str:
        """Load API key from environment variables."""
        load_dotenv()
        api_key = os.getenv('FIREWORKS_API_KEY')
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables")
        return api_key

    def get_completion(self, model: FireworksModel, prompt: str) -> Optional[str]:
        """Get completion from a specific model."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = model.to_dict()
        payload["messages"] = [{"role": "user", "content": prompt}]
        
        try:
            response = requests.post(self.BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Store response for analysis
            model.responses.append({
                "prompt": prompt,
                "response": result['choices'][0]['message']['content'],
                "timestamp": datetime.now().isoformat()
            })
            
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error making API call for model {model.model_id}: {e}")
            return None

    def compare_models(self, prompt: str) -> Dict[str, str]:
        """Compare responses from all models for the same prompt."""
        results = {}
        for model_name, model in self.models.items():
            print(f"\nTesting {model_name}...")
            response = self.get_completion(model, prompt)
            results[model_name] = response
        return results

def main():
    try:
        api = FireworksAPI()
        
        # Test prompts
        test_prompts = [
            "Write a haiku about artificial intelligence."
        ]
        
        # Test each prompt with all models
        for prompt in test_prompts:
            print(f"\n\n=== Testing prompt: {prompt} ===")
            results = api.compare_models(prompt)
            
            # Display results
            for model_name, response in results.items():
                print(f"\n{model_name.upper()} Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 