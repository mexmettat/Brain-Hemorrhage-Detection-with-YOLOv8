from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI(title="Brain Hemorrhage Detection API")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "runs/detect/baseline/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Falling back to pre-trained yolov8n.pt for demonstration if available.")
    model = YOLO("yolov8n.pt")

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        # Convert RGB to BGR for OpenCV/Ultralytics
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run inference with a lower confidence threshold to catch more anomalies
        results = model.predict(source=img_bgr, conf=0.10, save=False)
        result = results[0]

        # Extract detections
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]
                detections.append({
                    "class": class_name,
                    "confidence": round(conf * 100, 2)
                })

        # Plot annotated image
        annotated_img = result.plot()
        
        # Convert annotated image back to RGB
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Encode to Base64
        pil_img = Image.fromarray(annotated_img_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "image": img_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
