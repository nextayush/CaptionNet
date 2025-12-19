import sys
import logging
import traceback  # <--- NEW
from pathlib import Path

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import modules
try:
    from src import config
    from src.inference import CaptionGenerator
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import src modules. {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptionService")

_caption_generator = None

def load_ai_model():
    global _caption_generator
    if not config.FINAL_MODEL_PATH.exists():
        logger.error(f"❌ Model not found at {config.FINAL_MODEL_PATH}")
        raise FileNotFoundError("Model weights missing.")
        
    logger.info("Loading AI Model...")
    _caption_generator = CaptionGenerator()
    logger.info("✅ AI Model successfully loaded.")

def generate_caption(image_path: str, strategy: str = "beam"):
    global _caption_generator
    
    if _caption_generator is None:
        raise RuntimeError("AI Model is not loaded.")
    
    try:
        # --- CRITICAL: DEBUGGING PRINT ---
        print(f"DEBUG: Processing image at {image_path}")
        caption = _caption_generator.generate_caption(image_path, strategy=strategy)
        return caption
        
    except Exception as e:
        # --- THIS WILL PRINT THE EXACT ERROR ---
        print("\n" + "="*50)
        print("❌ PREDICTION CRASHED HERE:")
        traceback.print_exc()
        print("="*50 + "\n")
        raise e