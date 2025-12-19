import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

# Import config
try:
    from src import config
except ImportError:
    import config

def load_extraction_model():
    print("Loading VGG16 model...")
    # Load VGG16
    base_model = VGG16(weights='imagenet')
    
    # --- CRITICAL FIX ---
    # Instead of .pop(), we create a new model that outputs exactly what we want.
    # The 'fc2' layer is the second-to-last fully connected layer (4096 dimensions).
    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc2').output)
    
    print("VGG16 loaded. Output dimension: 4096.")
    return model

def extract_features(directory):
    # 1. Verify Directory
    if not os.path.exists(directory):
        print(f"❌ ERROR: Directory not found: {directory}")
        return {}

    # 2. List Files
    print(f"Scanning directory: {directory}")
    all_files = os.listdir(directory)
    valid_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Found {len(valid_images)} valid images.")

    if len(valid_images) == 0:
        print("❌ ERROR: No images found! Check your path in src/config.py.")
        return {}

    # 3. Load Model
    model = load_extraction_model()
    features = {}
    
    # 4. Extract
    print(f"Starting extraction on {len(valid_images)} images...")
    for name in tqdm(valid_images):
        filename = os.path.join(directory, name)
        try:
            # VGG16 expects 224x224
            image = load_img(filename, target_size=(224, 224), color_mode='rgb')
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image) # VGG16 specific preprocessing
            
            # Extract features
            feature = model.predict(image, verbose=0)
            
            # Get image ID
            image_id = os.path.splitext(name)[0]
            features[image_id] = feature
        except Exception as e:
            print(f"⚠️ Failed to process {name}: {e}")
            continue
        
    return features

if __name__ == "__main__":
    # Use the hardcoded path that worked for you earlier
    directory = r"D:\College\VIT\Sem_6\DL\CaptionNet\data\raw\images"

    features = extract_features(directory)
    
    if len(features) > 0:
        os.makedirs(os.path.dirname(config.FEATURES_DICT_PATH), exist_ok=True)
        with open(config.FEATURES_DICT_PATH, 'wb') as f:
            pickle.dump(features, f)
        print(f"🎉 Success! Saved features for {len(features)} images to {config.FEATURES_DICT_PATH}")
    else:
        print("❌ Extraction failed.")