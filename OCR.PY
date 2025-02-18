from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr")
async def perform_ocr(frame: UploadFile = File(...)):
    try:
        # Read and process image
        image_stream = io.BytesIO(await frame.read())
        image = Image.open(image_stream)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Clean up the text
        text = text.strip()
        
        return JSONResponse(content={
            "text": text if text else "No text detected"
        })
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)