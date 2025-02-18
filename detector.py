import cv2
import torch
from ultralytics import YOLO
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Initialize YOLO model
print("\nInitializing YOLO model...")
model = YOLO('yolov10x.pt')  # or whichever model you have
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Priority objects for visually impaired assistance
PRIORITY_OBJECTS = {
    'person': 'person',
    'car': 'vehicle',
    'truck': 'large vehicle',
    'stop sign': 'stop sign',
    'door': 'door',
    'stairs': 'stairs',
    'chair': 'chair',
    'bench': 'bench',
    'keyboard': 'keyboard',
    'laptop': 'laptop',
    'cell phone': 'cell phone',
    'bottle': 'bottle'

}

def get_position_description(box, frame_width):
    """Calculate relative position of detected object"""
    center_x = (box[0] + box[2]) / 2
    relative_x = center_x / frame_width
    
    if relative_x < 0.33:
        return "to your left"
    elif relative_x > 0.66:
        return "to your right"
    else:
        return "in front of you"

def estimate_distance(box_height, frame_height):
    """Estimate distance based on object size"""
    relative_height = box_height / frame_height
    if relative_height > 0.5:
        return "very close"
    elif relative_height > 0.3:
        return "close"
    elif relative_height > 0.1:
        return "moderate distance"
    else:
        return "far away"

@app.post("/detect")
async def detect(frame: UploadFile = File(...)):
    try:
        # Read and process image
        image_stream = io.BytesIO(await frame.read())
        image = Image.open(image_stream)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Run detection
        results = model.predict(frame, device=device, conf=0.4)
        detections = []
        boxes = []
        
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                if conf > 0.4:
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_height = y2 - y1
                    
                    # Get position and distance
                    position = get_position_description([x1, y1, x2, y2], frame_width)
                    distance = estimate_distance(box_height, frame_height)
                    
                    if class_name in PRIORITY_OBJECTS:
                        description = f"{PRIORITY_OBJECTS[class_name]} {position}, {distance}"
                        detections.append(description)
                        
                        boxes.append({
                            "coordinates": [x1, y1, x2, y2],
                            "label": PRIORITY_OBJECTS[class_name],
                            "confidence": conf
                        })
        
        return JSONResponse(content={
            "detections": detections,
            "boxes": boxes
        })
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)