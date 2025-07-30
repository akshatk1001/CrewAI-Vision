from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import base64
import litellm

class ImageToolInput(BaseModel):
    """Input schema for ImageTool."""
    image_path: str = Field(..., description="Path to the diagram image")
    prompt: str = Field(..., description="Prompt to describe what you want to analyze in the image")

class ImageTool(BaseTool):
    name: str = "anthropic_vision_tool"
    description: str = (
        "A tool that analyzes images using Claude 3.5 Sonnet vision model from Anthropic." \
        "It takes an image path and a prompt as input, and returns the model's response." \
        "The image is converted to a base64 data URL before being sent to the model."
    )
    args_schema: Type[BaseModel] = ImageToolInput

    def _run(self, image_path: str, prompt: str) -> str:
        print(f"Image path: {image_path}")
        print(f"Original prompt: {prompt}")
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode('utf-8')
                    
        # Create data URL for the image
        image_url = f"data:image/png;base64,{b64}"
                    
        # https://docs.litellm.ai/docs/providers/anthropic#usage---vision
        response = litellm.completion(
            model="anthropic/claude-3-5-sonnet-20240620",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
        )

        print(f"Anthropic vision model response received!")

        # Extract the response content
        response_text = response["choices"][0]["message"]["content"].strip()
        
        return response_text