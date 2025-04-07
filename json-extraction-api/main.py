from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
from io import BytesIO
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Loaded .env
load_dotenv() 

login(os.getenv("OCR_API"))

# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face OCR model & processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

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

        # Use Hugging Face TrOCR to extract text
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Try parsing the generated text as JSON
        parsed = json.loads(generated_text)

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
