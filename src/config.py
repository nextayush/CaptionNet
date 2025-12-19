import os
from pathlib import Path

# --- PROJECT ROOT ---
# Resolves to the parent of the 'src' folder
BASE_DIR = Path(__file__).resolve().parent.parent

# --- DATA PATHS ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Input paths (Flickr8k specifics)
IMAGES_DIR = RAW_DATA_DIR / "images"
CAPTION_FILE = RAW_DATA_DIR / "caption" / "Flickr8k.token.txt"

# Output paths
FEATURES_DICT_PATH = PROCESSED_DATA_DIR / "features.pkl"
DESCRIPTIONS_DICT_PATH = PROCESSED_DATA_DIR / "descriptions.txt"
TOKENIZER_PATH = PROCESSED_DATA_DIR / "tokenizer.pkl"

# --- MODEL ARTIFACTS ---
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.h5"

# --- HYPERPARAMETERS ---
# Image processing
IMG_SIZE = (224, 224)   # Standard for VGG16/ResNet
IMG_SHAPE = (224, 224, 3)

# Model Architecture
VOCAB_SIZE = None       # Will be set dynamically after preprocessing
MAX_LENGTH = None       # Will be set dynamically (usually around 34 for Flickr8k)
EMBEDDING_DIM = 256     # Size of word vectors
UNITS = 256             # LSTM units
DROPOUT = 0.5           # Regularization rate

# Training
BATCH_SIZE = 32         # Reduce to 16 if you run out of memory
EPOCHS = 20
LEARNING_RATE = 0.001

# --- UTILS ---
def make_directories():
    """Ensure all necessary directories exist before running scripts."""
    dirs = [PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINT_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    make_directories()
    print(f"Project configuration loaded. Root: {BASE_DIR}")