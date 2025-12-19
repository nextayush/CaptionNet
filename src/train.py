import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Import local modules
import config
from data_loader import data_generator
from model_builder import define_model
from preprocess_text import load_doc, load_descriptions, clean_descriptions

def load_set_of_image_ids(filename):
    """
    Loads a set of image IDs from a file (e.g., Flickr_8k.trainImages.txt).
    """
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def filter_clean_descriptions(all_descriptions, dataset_ids):
    """
    Returns a dictionary of descriptions only for the images in dataset_ids.
    Adds <start> and <end> tokens if they aren't already there.
    """
    clean_ds = dict()
    for key, desc_list in all_descriptions.items():
        if key in dataset_ids:
            clean_ds[key] = desc_list
    return clean_ds

def load_photo_features(filename, dataset_ids):
    """
    Loads the entire features dictionary, then filters for the specific dataset.
    """
    with open(filename, 'rb') as f:
        all_features = pickle.load(f)
    
    features = {k: all_features[k] for k in dataset_ids if k in all_features}
    return features

def train():
    print("--- 1. Loading Data & Configurations ---")
    
    # Load Tokenizer
    with open(config.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    # We can hardcode this if known, or calculate it. 34 is standard for Flickr8k.
    max_length = config.MAX_LENGTH if config.MAX_LENGTH else 34
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Max Length: {max_length}")

    # Load Training Image IDs (Flickr_8k.trainImages.txt)
    # Note: Ensure this file exists in your raw data folder or adjust path
    train_split_file = config.RAW_DATA_DIR / "caption" / "Flickr_8k.trainImages.txt"
    train_ids = load_set_of_image_ids(train_split_file)
    print(f"Training Images: {len(train_ids)}")

    # Load all descriptions, then filter for training
    # We re-load raw text to ensure we match the IDs correctly
    doc = load_doc(config.CAPTION_FILE)
    descriptions = load_descriptions(doc)
    clean_descriptions(descriptions) # Clean them
    train_descriptions = filter_clean_descriptions(descriptions, train_ids)

    # Load Features
    train_features = load_photo_features(config.FEATURES_DICT_PATH, train_ids)
    print(f"Loaded {len(train_features)} feature vectors.")

    
    print("--- 2. Building Model ---")
    model = define_model(vocab_size, max_length)
    
    # Define Checkpoints
    # Save the model whenever 'loss' improves (lowers)
    filepath = str(config.CHECKPOINT_DIR / "model-ep{epoch:03d}-loss{loss:.3f}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    # Reduce Learning Rate if loss stops improving
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    
    callbacks_list = [checkpoint, reduce_lr]


    print("--- 3. Starting Training ---")
    # Create the data generator
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size, config.BATCH_SIZE)
    
    # Calculate steps per epoch (Total Samples / Batch Size)
    steps = len(train_descriptions) // config.BATCH_SIZE

    try:
        model.fit(
            generator,
            epochs=config.EPOCHS,
            steps_per_epoch=steps,
            callbacks=callbacks_list,
            verbose=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        
    # Save Final Model
    print(f"Saving final model to {config.FINAL_MODEL_PATH}")
    model.save(config.FINAL_MODEL_PATH)

if __name__ == "__main__":
    train()