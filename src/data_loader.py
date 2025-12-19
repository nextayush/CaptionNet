import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    """
    Creates input-output sequence pairs for a single image.
    
    Args:
        tokenizer: Fitted Keras Tokenizer.
        max_length: The defined maximum sequence length.
        desc_list: List of caption strings for this specific image.
        photo: The 4096-dim feature vector for this image.
        vocab_size: Size of the vocabulary.
        
    Returns:
        X1: List of image vectors.
        X2: List of input text sequences.
        y: List of output target words (one-hot encoded).
    """
    X1, X2, y = list(), list(), list()
    
    # Walk through each description for the image
    for desc in desc_list:
        # Encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        
        # Split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # Split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            
            # Pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            
            # Encode output sequence (One-Hot Encoding)
            # This creates a sparse vector the size of the vocab
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            
            # Store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
            
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size=32):
    keys = list(descriptions.keys())
    print(f"DEBUG: Generator started. Total keys: {len(keys)}")
    print(f"DEBUG: Sample Photo Key: {list(photos.keys())[0] if photos else 'EMPTY'}")
    
    while True:
        count = 0
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i : i + batch_size]
            input_imgs, input_seqs, output_words = [], [], []
            
            for key in batch_keys:
                # 1. Check description
                desc_list = descriptions.get(key)
                
                # 2. Check Feature
                photo = photos.get(key)
                
                if photo is None:
                    # This is the common cause of infinite loops!
                    # If keys don't match, we skip. If ALL skip, we hang.
                    continue
                
                # Flatten if needed (for ResNet/VGG)
                photo = np.array(photo).flatten()
                
                in_img, in_seq, out_word = create_sequences(
                    tokenizer, max_length, desc_list, photo, vocab_size
                )
                
                for k in range(len(in_img)):
                    input_imgs.append(in_img[k])
                    input_seqs.append(in_seq[k])
                    output_words.append(out_word[k])
            
            if len(input_imgs) > 0:
                count += 1
                # Only print the first batch to confirm it works
                if count == 1:
                    print(f"DEBUG: Yielding first batch of size {len(input_imgs)}")
                    
                yield (
                    {
                        'image_input': np.array(input_imgs), 
                        'text_input': np.array(input_seqs)
                    }, 
                    np.array(output_words)
                )
            else:
                 # If we found NO valid data in this batch, warn the user
                 print(f"WARNING: Batch {i} was empty. Mismatch between photos and captions?")