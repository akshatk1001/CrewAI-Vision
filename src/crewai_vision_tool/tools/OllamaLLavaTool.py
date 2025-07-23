from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import base64
import litellm

class OllamaLLavaToolInput(BaseModel):
    """Input schema for OllamaLLavaTool."""
    image_path: str = Field(..., description="Path to the diagram image")
    prompt: str = Field(..., description="Prompt to describe what you want to analyze in the image")

class OllamaLLavaTool(BaseTool):
    name: str = "ollama_vision_tool"
    description: str = (
        "Analyze images using Ollama's LLaVA vision model. "
        "Provide an image path and a prompt describing what you want to analyze."
    )
    args_schema: Type[BaseModel] = OllamaLLavaToolInput

    def _run(self, image_path: str, prompt: str) -> str:
        print(f"Image path: {image_path}")
        print(f"Prompt: {prompt}")
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode('utf-8')
                    
        # Create data URL for the image
        image_url = f"data:image/jpeg;base64,{b64}"
                    
        # Use LiteLLM to call Ollama's LLaVA model
        # https://docs.litellm.ai/docs/providers/ollama#ollama-vision-models
        response = litellm.completion(
            model="ollama/llama3.2-vision:latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            api_base="http://localhost:11434"
        )
        
        print(f"Ollama LLaVA response received!")
        
        # Extract the response content
        response_text = response.choices[0].message.content.strip()
        
        return response_text