
# CaptionNet 

CaptionNet is an end‑to‑end image captioning system that extracts visual features from images using a VGG16‑based encoder and generates natural‑language descriptions with a neural decoder. 

---

## Features

- **VGG16‑based encoder** for extracting high‑level visual features from input images.   
- Sequence decoder with text preprocessing and vocabulary handling for caption generation.   
- Modular training pipeline with separate scripts for data loading, feature extraction, model building, and training.   
- REST API backend built with FastAPI for serving caption generation as an HTTP service.   
- Frontend folder prepared for a web UI to upload images and display generated captions. 

---

## Project Structure

```
CaptionNet/
├── backend/              # FastAPI service, API routes, model loading
├── frontend/             # Web UI (image upload + caption display)
├── models/               # Saved weights/checkpoints for the caption model
├── src/
│   ├── __init__.py       # Package initializer
│   ├── config.py         # Paths, hyperparameters, and global config
│   ├── data_loader.py    # Dataset loading and batching
│   ├── extract_features.py# VGG16 feature extraction for images
│   ├── inference.py      # Inference utilities for caption generation
│   ├── model_builder.py  # Encoder–decoder model definition
│   ├── preprocess_text.py# Tokenization, vocabulary, and text cleaning
│   └── train.py          # Training loop and model optimization
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # Project documentation
```


---

## Installation

1. **Clone the repository**

   ```
   git clone https://github.com/nextayush/CaptionNet.git
   cd CaptionNet
   ```

2. **Create and activate a virtual environment (recommended)**

   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```
   The project uses TensorFlow 2.15, NumPy, Pillow, NLTK, scikit‑learn, FastAPI, Uvicorn, Pydantic, and related tools. 

---

## Usage

### 1. Prepare data and extract features

- Place your image dataset and caption annotations according to the paths configured in `src/config.py`.   
- Run feature extraction using the VGG16 encoder:

  ```
  python -m src.extract_features
  ```

This will compute and store image feature vectors used during training. 

### 2. Preprocess captions

- Clean and tokenize captions, build vocabulary, and save processed sequences:

  ```
  python -m src.preprocess_text
  ```

This step prepares text data for the decoder model. 

### 3. Train the model

- Start model training with:

  ```
  python -m src.train
  ```

The script will load extracted features and processed captions, build the encoder–decoder model, and save trained weights under `models/`. 

### 4. Run inference from Python

- Use `src.inference.py` to generate captions for new images:

  ```
  python -m src.inference --image_path path/to/image.jpg
  ```

The script will load the trained model from `models/` and print the generated caption. 

---

## API Server

The backend provides an HTTP interface for caption generation using FastAPI. 

1. **Start the API server**

   From the project root:

   ```
   cd backend
   uvicorn main:app --reload
   ```

   Adjust the module name (`main:app`) if your FastAPI entrypoint file is named differently. 

2. **Example request**

   - Endpoint (typical): `POST /caption` with a multipart form containing an image file.   
   - Use `curl` or any HTTP client:

     ```
     curl -X POST "http://localhost:8000/caption" \
       -F "file=@path/to/image.jpg"
     ```

   The API responds with a JSON object containing the generated caption. 

---

## Frontend

The `frontend/` directory is intended for a web interface that interacts with the FastAPI backend. 

Typical flow:

- User uploads an image via the web UI.   
- Frontend sends the file to the backend caption API and displays the generated description. 

Integrate or build your preferred stack (e.g., Vanilla JS, React, or a template engine) inside this folder.

---

## Requirements

Key dependencies (see `requirements.txt` for full versions): 

- **Core ML:** TensorFlow, NumPy, scikit‑learn, Pandas.   
- **Image & text:** Pillow, NLTK, tqdm.   
- **Serving:** FastAPI, Uvicorn, python‑multipart, Pydantic.   
- **Development:** Black, Flake8, Jupyter. 

---
