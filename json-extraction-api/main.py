from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
from io import BytesIO
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Loaded .env
load_dotenv() 

login(os.getenv("OCR_API"))

app = FastAPI()

# Load Callisto-OCR3-2B-Instruct model and processor
processor = AutoProcessor.from_pretrained("prithivMLmods/Callisto-OCR3-2B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("prithivMLmods/Callisto-OCR3-2B-Instruct").eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input schema
class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/")
async def extract_json(image_request: ImageRequest):
    try:
        # Decode base64 image
        image_data = image_request.imageBase64.split(",")[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Build instruction prompt (as per model's expected input)
        prompt = "Extract and return the data in JSON format from this image."

        # Preprocess image and prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        # Generate output from model
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Attempt to parse as JSON
        parsed = json.loads(generated_text)

        return {
            "success": True,
            "data": parsed,
            "message": "Successfully extracted JSON from image"
        }

    except json.JSONDecodeError:
        return {
            "success": False,
            "data": generated_text,
            "message": "OCR worked, but couldn't parse valid JSON."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
