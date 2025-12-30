# Plant Disease Detection System

This project is a comprehensive Deep Learning solution designed to detect and classify plant diseases from leaf images. It consists of a training pipeline using TensorFlow/Keras and a deployment API for serving predictions.

## 📂 Project Overview

The project is divided into two main components:

1.  **Model Training (`Train_plant_disease_model.ipynb`)**: A Jupyter Notebook that handles data ingestion, augmentation, and training of a Convolutional Neural Network (CNN).
2.  **Inference API (`plant-api` Repo)**: A backend service that serves the trained model to make real-time predictions.

## 🧠 Part 1: Model Training

The model is trained using the **New Plant Diseases Dataset**.

### Dataset Details

- **Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: The model is trained to recognize **38 unique classes**, including healthy and diseased states of Apple, Corn, Grape, Potato, Tomato, and more.
- **Image Size**: Images are resized to `128x128` pixels.

### Training Pipeline

1.  **Data Loading & Preprocessing**:
    - Dataset is downloaded using `opendatasets`.
    - Images are loaded into training and validation splits with a batch size of 32.
2.  **Data Augmentation**:
    - To improve generalization, the following augmentations are applied: `RandomFlip`, `RandomRotation` (0.1), and `RandomContrast` (0.1).
3.  **Model Architecture**:
    - A custom **CNN architecture** is used, consisting of 5 blocks of `Conv2D`, `BatchNormalization`, `ReLU`, and `MaxPooling2D` layers.
    - The head of the network includes a `Flatten` layer, a `Dense` layer (1024 units), and a final `Dense` output layer with **Softmax** activation for 38-class classification.
4.  **Artifacts**:
    - **Model**: Saved as `plant_disease_classification_v3.keras`.
    - **Class Labels**: Saved as `class.json`.

---

## 🚀 Part 2: API Deployment

The deployment code is hosted in the [plant-api repository](https://github.com/biswajr/plant-api).

### Prerequisites

- Python 3.x
- TensorFlow
- FastAPI (or the framework defined in the repo)
- Pillow, NumPy

### Setup & Installation

1.  **Clone the API repository**:

    ```bash
    git clone [https://github.com/biswajr/plant-api.git](https://github.com/biswajr/plant-api.git)
    cd plant-api
    ```

2.  **Prepare Model Artifacts**:

    - Copy the trained model `plant_disease_classification_v3.keras` from the training output to the API directory.
    - Copy the `class.json` file to the API directory.

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Server**:
    _(Assuming FastAPI/Uvicorn)_
    ```bash
    uvicorn main:app --reload
    or
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 3000
    ```

### API Usage

**Endpoint**: `/predict`
**Method**: `POST`

**Request**:
Upload an image file via `multipart/form-data`.

**Example using cURL**:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_leaf.jpg;type=image/jpeg'
```

## 🤝 Contributing

Feel free to open issues or submit pull requests if you have suggestions for improving the model architecture or API performance.

## 📜 License

This project is open-source. Please check the specific license files in the repository for details.
