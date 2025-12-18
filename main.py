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

# for serverless
BASE_DIR = Path("/tmp")
MODEL_PATH = BASE_DIR / "model.keras"
CLASS_PATH = BASE_DIR / "class.json"

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "model.keras"
# CLASS_PATH = BASE_DIR / "class.json"

TARGET_SIZE = (128, 128)

cnn = None
CLASS_NAMES = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model_and_class():
    global cnn, CLASS_NAMES
    cnn = None
    CLASS_NAMES = None

    if not MODEL_PATH.is_file():
        return
    if not CLASS_PATH.is_file():
        return

    try:
        cnn = tf.keras.models.load_model(str(MODEL_PATH))
        with open(CLASS_PATH, "r") as f:
            CLASS_NAMES = json.load(f)
        if not isinstance(CLASS_NAMES, list):
            cnn = None
            CLASS_NAMES = None
    except:
        cnn = None
        CLASS_NAMES = None

@app.on_event("startup")
def startup_event():
    load_model_and_class()

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    img_io = io.BytesIO(image_bytes)
    image = tf.keras.preprocessing.image.load_img(
        img_io, target_size=TARGET_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(arr, axis=0)

@app.get("/", response_class=HTMLResponse)
def index():
    return "Plant Disease API is running"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if cnn is None or CLASS_NAMES is None:
        raise HTTPException(503, "Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file")

    input_arr = preprocess_image_bytes(image_bytes)
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
    <html>
    <body>
      <form action="/upload-model" method="post" enctype="multipart/form-data">
        <input type="file" name="model" accept=".keras" required><br><br>
        <input type="file" name="class_file" accept=".json" required><br><br>
        <button>Upload</button>
      </form>
    </body>
    </html>
    """

@app.post("/upload-model")
async def upload_model(
    model: UploadFile = File(...),
    class_file: UploadFile = File(...)
):
    if not model.filename.endswith(".keras"):
        raise HTTPException(400, "Model must be .keras")

    if not class_file.filename.endswith(".json"):
        raise HTTPException(400, "Class must be .json")

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if CLASS_PATH.exists():
        CLASS_PATH.unlink()

    with open(MODEL_PATH, "wb") as f:
        f.write(await model.read())

    with open(CLASS_PATH, "wb") as f:
        f.write(await class_file.read())

    load_model_and_class()

    if cnn is None or CLASS_NAMES is None:
        raise HTTPException(500, "Model reload failed")

    return {
        "status": "success",
        "message": "Model and class uploaded & loaded successfully",
    }
