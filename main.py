import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.keras"))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", "class.json"))
TARGET_SIZE = (128, 128)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

cnn = None
CLASS_NAMES = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model_and_classes():
    global cnn, CLASS_NAMES

    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        cnn = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            CLASS_NAMES = json.load(f)
        print("Model & classes loaded")
    else:
        cnn = None
        CLASS_NAMES = None
        print("Model or class.json missing")

@app.on_event("startup")
def startup_event():
    load_model_and_classes()

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img_io = io.BytesIO(image_bytes)
    image = tf.keras.preprocessing.image.load_img(
        img_io,
        target_size=TARGET_SIZE
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if cnn is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        input_arr = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    predictions = cnn.predict(input_arr)
    result_index = int(np.argmax(predictions))
    confidence = int(round(float(predictions[0][result_index]) * 100))

    if confidence < 60:
        return {
            "class_index": -1,
            "class_name": "Unknown / No leaf detected",
            "confidence": confidence
        }

    return {
        "class_index": result_index,
        "class_name": CLASS_NAMES[result_index],
        "confidence": confidence
    }


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Upload Model</title></head>
    <body style="font-family: Arial; padding: 40px;">
        <h2>Upload Model & Classes</h2>
        <form action="/upload-model" method="post" enctype="multipart/form-data">
            <label>Model (.keras)</label><br>
            <input type="file" name="model" accept=".keras" required><br><br>

            <label>Classes (classes.json)</label><br>
            <input type="file" name="classes" accept=".json" required><br><br>

            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """


@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    classes: UploadFile = File(...)
):
    if not model.filename.endswith(".keras"):
        raise HTTPException(status_code=400, detail="Model must be .keras")

    if not classes.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Classes must be .json")

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if CLASSES_PATH.exists():
        CLASSES_PATH.unlink()

    with open(MODEL_PATH, "wb") as f:
        f.write(await model.read())

    with open(CLASSES_PATH, "wb") as f:
        f.write(await classes.read())

    load_model_and_classes()

    return {
        "status": "success",
        "message": "Model and classes uploaded and reloaded successfully"
    }
