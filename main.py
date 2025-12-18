import os
import io
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json

# ================== Config & Init ==================

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.keras"))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", "classes.json"))

TARGET_SIZE = (128, 128)  # must match training

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

cnn = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Helpers ==================

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img_io = io.BytesIO(image_bytes)

    image = tf.keras.preprocessing.image.load_img(
        img_io,
        target_size=TARGET_SIZE
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # (1, H, W, C)

    # IMPORTANT: match your training preprocessing
    # If in notebook you did input_arr = input_arr / 255.0
    # uncomment this:
    # input_arr = input_arr / 255.0

    return input_arr

# ================== Routes ==================

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Send an image in form-data with field name 'file'.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        input_arr = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    predictions = cnn.predict(input_arr)
    probs = tf.nn.softmax(predictions[0]).numpy()

    result_index = int(np.argmax(probs))
    model_prediction = CLASS_NAMES[result_index]

    confidence_score = probs[result_index] * 100.0
    confidence_int = int(round(confidence_score))

    return {
        "class_index": result_index,
        "class_name": model_prediction,
        "confidence": confidence_int,   # e.g. 87
        "all_probs": probs.tolist(),
    }
