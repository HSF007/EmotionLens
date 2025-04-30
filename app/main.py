import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from starlette.responses import Response
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_emotion

app = FastAPI(title="EmotionLens API", description="API for emotion detection from facial images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount the static directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

PREDICTION_COUNT = Counter('emotion_predictions_total', 'Total number of emotion predictions')
PREDICTION_LATENCY = Histogram('emotion_prediction_latency_seconds', 'Latency of emotion predictions (seconds)')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@app.get("/")
async def root():
    return {"message": "Welcome to EmotionLens API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"Failed to decode image: {file.filename}")
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)
        
        result = predict_emotion(img)
        
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        logger.info(f"Prediction: {result['emotion']} with confidence {result['confidence']:.2f}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)