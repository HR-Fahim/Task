from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import json
import os
import re
from dotenv import load_dotenv
import requests
from PIL import Image, ImageEnhance
from io import BytesIO

# Load .env
load_dotenv()

# API & Site details
OPENROUTER_API_KEY = os.getenv("API_KEY")
REFERER = os.getenv("REFERER") or "http://localhost"
SITE_NAME = os.getenv("SITE_NAME") or "JSON Extraction App"

# FastAPI app
app = FastAPI()

# Input model
class ImageRequest(BaseModel):
    imageBase64: str

# Clean and format values
def clean_value(value: str) -> str:
    if not value:
        return None
    return value.strip(" \":,")  # Strips unwanted chars from both ends

def format_response_to_json(text: str):
    # Try JSON parsing first
    try:
        parsed = json.loads(text)
        return {
            "name": clean_value(parsed.get("name")),
            "organization": clean_value(parsed.get("organization")),
            "address": clean_value(parsed.get("address")),
            "mobile": clean_value(parsed.get("mobile")),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: regex-based extraction
    name_match = re.search(r"(?:Name|Full Name)[^\w]*[:\-]?\s*(.*)", text, re.IGNORECASE)
    org_match = re.search(r"(?:Organization|Company|Institute)[^\w]*[:\-]?\s*(.*)", text, re.IGNORECASE)
    address_match = re.search(r"(?:Address)[^\w]*[:\-]?\s*(.*)", text, re.IGNORECASE)
    mobile_match = re.search(r"(?:Mobile|Phone|Tel|Contact)[^\w]*[:\-]?\s*(.*)", text, re.IGNORECASE)

    return {
        "name": clean_value(name_match.group(1)) if name_match else None,
        "organization": clean_value(org_match.group(1)) if org_match else None,
        "address": clean_value(address_match.group(1)) if address_match else None,
        "mobile": clean_value(mobile_match.group(1)) if mobile_match else None,
    }

# Convert transparent images to white background with better handling
def handle_transparency(base64_image: str) -> str:
    # Remove the data prefix if present
    header_removed = base64_image.split(",")[-1]
    image_bytes = base64.b64decode(header_removed)
    image = Image.open(BytesIO(image_bytes))

    # Ensure the image is in RGBA mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Prepare a white background
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Composite the image on top of the white background using the alpha channel as the mask
    image = Image.alpha_composite(background, image)

    # Convert to RGB to remove the alpha channel
    image = image.convert("RGB")

    # Optionally enhance contrast and sharpness to help with text detection
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast

    # Sharpen the image to improve text clarity
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(2)  # Sharpen the image

    # Save as PNG again
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + processed_base64

@app.post("/")
def extract_json(image_request: ImageRequest):
    try:
        image_data = handle_transparency(image_request.imageBase64.strip())

        # LLaMA 4 request
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": REFERER,
                "X-Title": SITE_NAME
            },
            data=json.dumps({
                "model": "meta-llama/llama-4-maverick:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the following information from the image and return as JSON:\n- name\n- organization\n- address\n- mobile"
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

        formatted_data = format_response_to_json(llama_text)

        return {
            "success": True,
            "data": formatted_data,
            "message": "Successfully extracted structured JSON from image and I have used LLAMA 4 for text extraction"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
