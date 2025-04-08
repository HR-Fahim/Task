from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import os
import re
from dotenv import load_dotenv
import requests
from PIL import Image, ImageEnhance
from io import BytesIO

# Load environment variables from .env
load_dotenv()

# API key from .env
OPENROUTER_API_KEY = os.getenv("API_KEY")

# FastAPI app initialization
app = FastAPI()

# CORS Middleware Configuration
origins = [
    "https://json-extraction-challenge.intellixio.com",  # Add your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Allow your frontend
    allow_credentials=True,
    allow_methods=["*"],               # Allow all HTTP methods (POST, OPTIONS, etc.)
    allow_headers=["*"],              # Allow all headers including Content-Type
)

# Input model for request body
class ImageRequest(BaseModel):
    imageBase64: str

# Convert transparent images to white background and enhance
def handle_transparency(base64_image: str) -> str:
    header_removed = base64_image.split(",")[-1]
    image_bytes = base64.b64decode(header_removed)
    image = Image.open(BytesIO(image_bytes))

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image)
    image = image.convert("RGB")

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(2)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + processed_base64

# Endpoint to handle JSON extraction
@app.post("/")
def extract_json(image_request: ImageRequest):
    try:
        image_data = handle_transparency(image_request.imageBase64.strip())

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-4-maverick:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Only extract the text and show in exact JSON format. Do not use words like 'json' or symbols like ```"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            }
                        ]
                    }
                ]
            })
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        llama_text = result["choices"][0]["message"]["content"]

        formatted_data = json.loads(llama_text)

        return {
            "success": True,
            "data": formatted_data,
            "message": "Successfully extracted structured JSON from image and I have used LLAMA 4 for text extraction"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
