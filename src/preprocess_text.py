import string
import pickle
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# Import configuration
import config

def load_doc(filename):
    """Helper to read a text file and return string."""
    try:
        with open(filename, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}. Check config.py paths.")

def load_descriptions(doc):
    """
    Parses the raw Flickr8k text format.
    Format in file: image_id#0 caption...
    Returns: Dict {image_id: [caption1, caption2, ...]}
    """
    mapping = collections.defaultdict(list)
    
    for line in doc.split('\n'):
        # Split line by white space
        tokens = line.split()
        if len(tokens) < 2:
            continue
            
        # Take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        
        # Remove filename extension from image_id (e.g. '1000268201_693b08cb0e.jpg#0')
        image_id = image_id.split('.')[0]
        
        # Convert description back to string
        image_desc = ' '.join(image_desc)
        
        mapping[image_id].append(image_desc)
        
    print(f"Loaded {len(mapping)} images with captions.")
    return mapping

def clean_descriptions(descriptions):
    """
    Performs text cleaning: lowercasing, punctuation removal, 
    removing numbers, and adding <start>/<end> tokens.
    """
    # Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    
    for key, desc_list in tqdm(descriptions.items(), desc="Cleaning Captions"):
        for i in range(len(desc_list)):
            desc = desc_list[i]
            
            # 1. Tokenize
            desc = desc.split()
            
            # 2. Convert to lower case
            desc = [word.lower() for word in desc]
            
            # 3. Remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            
            # 4. Remove 's' and 'a' hanging letters (optional, but standard for Flickr8k)
            desc = [word for word in desc if len(word) > 1]
            
            # 5. Remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            
            # 6. Add <start> and <end> tokens
            # CRITICAL: This teaches the model where to start and stop
            desc = ['startseq'] + desc + ['endseq']
            
            # Store as string
            desc_list[i] =  ' '.join(desc)

    return descriptions

def save_descriptions(descriptions, filename):
    """Saves the cleaned descriptions to a text file (one per line)."""
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)

def create_tokenizer(descriptions):
    """
    Fits a Keras Tokenizer on the cleaned text.
    Returns: The fitted Tokenizer object.
    """
    # Collect all captions into a single list
    all_captions = []
    for key in descriptions.keys():
        [all_captions.append(d) for d in descriptions[key]]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    return tokenizer

def get_max_length(descriptions):
    """Calculates the longest caption in words."""
    all_captions = []
    for key in descriptions.keys():
        [all_captions.append(d) for d in descriptions[key]]
    return max(len(d.split()) for d in all_captions)

if __name__ == "__main__":
    # 1. Load raw text
    print(f"Loading captions from: {config.CAPTION_FILE}")
    doc = load_doc(config.CAPTION_FILE)
    
    # 2. Parse into dictionary
    descriptions = load_descriptions(doc)
    
    # 3. Clean
    cleaned_descriptions = clean_descriptions(descriptions)
    
    # 4. Save cleaned text to disk (useful for debugging)
    print(f"Saving cleaned descriptions to {config.DESCRIPTIONS_DICT_PATH}...")
    save_descriptions(cleaned_descriptions, config.DESCRIPTIONS_DICT_PATH)
    
    # 5. Build Tokenizer
    print("Building Tokenizer...")
    tokenizer = create_tokenizer(cleaned_descriptions)
    
    # 6. Save Tokenizer
    print(f"Saving Tokenizer to {config.TOKENIZER_PATH}...")
    with open(config.TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 7. Output Statistics
    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(cleaned_descriptions)
    
    print("-" * 30)
    print(f"Preprocessing Complete.")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Description Length: {max_length}")
    print("-" * 30)
    print("ACTION REQUIRED: Update 'src/config.py' if you want to hardcode these values.")
    print("-" * 30)