import io
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np

load_dotenv()

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "model.keras")
CLASSES_PATH = BASE_DIR / os.getenv("CLASSES_PATH", "classes.json")

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

    cnn = None
    CLASS_NAMES = None

    if not MODEL_PATH.exists():
        print("model.keras not found, skipping load")
        return

    if not CLASSES_PATH.exists():
        print("classes.json not found, skipping load")
        return

    try:
        cnn = tf.keras.models.load_model(str(MODEL_PATH))
        with open(CLASSES_PATH, "r") as f:
            CLASS_NAMES = json.load(f)

        if not isinstance(CLASS_NAMES, list):
            raise ValueError("classes.json must be a list")

        print("Model & classes loaded successfully")

    except Exception as e:
        cnn = None
        CLASS_NAMES = None
        print(f"Model load failed: {e}")

@app.on_event("startup")
def startup_event():
    load_model_and_classes()

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img_io = io.BytesIO(image_bytes)
    image = tf.keras.preprocessing.image.load_img(
        img_io, target_size=TARGET_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(arr, axis=0)

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

    preds = cnn.predict(input_arr)
    idx = int(np.argmax(preds))
    confidence = int(round(float(preds[0][idx]) * 100))

    if confidence < 60:
        return {
            "class_index": -1,
            "class_name": "Unknown / No leaf detected",
            "confidence": confidence,
        }

    return {
        "class_index": idx,
        "class_name": CLASS_NAMES[idx],
        "confidence": confidence,
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
    try:
        if not model.filename.endswith(".keras"):
            raise HTTPException(400, "Model must be .keras")

        if not classes.filename.endswith(".json"):
            raise HTTPException(400, "Classes must be .json")

        with open(MODEL_PATH, "wb") as f:
            f.write(await model.read())

        with open(CLASSES_PATH, "wb") as f:
            f.write(await classes.read())

        load_model_and_classes()

        if cnn is None or CLASS_NAMES is None:
            raise RuntimeError("Model reload failed")

        return {
            "status": "success",
            "message": "Model and classes uploaded & loaded successfully"
        }

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))