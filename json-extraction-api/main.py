from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
from io import BytesIO
import pytesseract
import json
import os
from dotenv import load_dotenv

# Loaded .env
load_dotenv() 

api_key = os.getenv("OCR_API")

# Initialize FastAPI app
app = FastAPI()

# Input model
class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/")
async def extract_json(image_request: ImageRequest):
    try:
        # Decode the base64 image
        image_data = image_request.imageBase64.split(",")[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(image)

        # Try parsing the extracted text as JSON
        try:
            parsed = json.loads(extracted_text)
            return {
                "success": True,
                "data": parsed,
                "message": "Successfully extracted JSON from image"
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "data": None,
                "message": "OCR worked, but couldn't parse valid JSON."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
