from pathlib import Path
import io

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray pneumonia classification using MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "pneumonia_mobilenet.keras"

model = tf.keras.models.load_model(MODEL_PATH)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png"
}


@app.get("/")
def root():
    return {
        "message": "Pneumonia Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "MobileNetV2"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "MobileNetV2",
        "input_size": "224x224",
        "classes": ["NORMAL", "PNEUMONIA"]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are allowed"
        )

    image_data = await file.read()

    try:
        image = Image.open(io.BytesIO(image_data))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file"
        )

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    prediction = "PNEUMONIA" if probability >= 0.5 else "NORMAL"

    confidence = (
        probability
        if prediction == "PNEUMONIA"
        else 1 - probability
    )

    return {
        "filename": file.filename,
        "prediction": prediction,
        "probability": round(probability, 4),
        "confidence": round(confidence, 4)
    }