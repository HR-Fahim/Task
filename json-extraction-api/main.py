from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import json

app = FastAPI()

class ImageRequest(BaseModel):
    imageBase64: str

@app.post("/")
async def extract_json(image_request: ImageRequest):
    try:
        # Clean base64 string
        image_data = image_request.imageBase64.split(",")[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Extract text from image
        extracted_text = pytesseract.image_to_string(image)

        # Parse JSON
        extracted_json = json.loads(extracted_text)

        return {
            "success": True,
            "data": extracted_json,
            "message": "Successfully extracted JSON from image"
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Unable to parse JSON from image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
