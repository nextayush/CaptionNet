import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Import config and model builder
try:
    from src import config
    from src.model_builder import define_model
except ImportError:
    import config
    from model_builder import define_model

class CaptionGenerator:
    def __init__(self):
        print("--- Loading Caption Generator (VGG16) ---")
        
        # 1. Load Tokenizer
        print(f"Loading Tokenizer from {config.TOKENIZER_PATH}...")
        with open(config.TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = config.MAX_LENGTH if config.MAX_LENGTH else 34
        
        # 2. Rebuild & Load Weights
        print(f"Building Model Architecture (Vocab: {self.vocab_size}, MaxLen: {self.max_length})...")
        try:
            self.model = define_model(self.vocab_size, self.max_length)
            print(f"Loading weights from {config.FINAL_MODEL_PATH}...")
            self.model.load_weights(config.FINAL_MODEL_PATH)
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading model: {e}")
            raise e
        
        # 3. Load VGG16 Feature Extractor
        print("Loading VGG16 Feature Extractor...")
        vgg_model = VGG16(weights='imagenet')
        # Explicitly get the 'fc2' layer (4096 dim)
        self.vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.get_layer('fc2').output)
        
        print("--- Caption Generator Ready ---")

    def extract_features(self, image_path):
        # VGG expects 224x224
        image = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = self.vgg_model.predict(image, verbose=0)
        return feature

    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_caption(self, image_path, strategy='beam', k=3):
        photo = self.extract_features(image_path)
        if strategy == 'greedy':
            return self._greedy_search(photo)
        else:
            return self._beam_search(photo, k)

    def _greedy_search(self, photo):
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.word_for_id(yhat)
            if word is None: break
            in_text += ' ' + word
            if word == 'endseq': break
        return in_text.replace('startseq', '').replace('endseq', '').strip()

    def _beam_search(self, photo, k=3):
        start_seq = self.tokenizer.texts_to_sequences(['startseq'])[0]
        sequences = [[start_seq, 0.0]]
        
        while len(sequences[0][0]) < self.max_length:
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == self.tokenizer.word_index.get('endseq'):
                    all_candidates.append([seq, score])
                    continue
                padded_seq = pad_sequences([seq], maxlen=self.max_length)
                yhat = self.model.predict([photo, padded_seq], verbose=0)[0]
                top_k_indices = np.argsort(yhat)[-k:]
                for word_index in top_k_indices:
                    new_score = score + np.log(yhat[word_index] + 1e-10)
                    new_seq = seq + [word_index]
                    all_candidates.append([new_seq, new_score])
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:k]
            if sequences[0][0][-1] == self.tokenizer.word_index.get('endseq') and len(sequences[0][0]) > 1:
                break
        
        best_seq = sequences[0][0]
        final_caption = []
        for i in best_seq:
            word = self.word_for_id(i)
            if word and word not in ['startseq', 'endseq']:
                final_caption.append(word)
        return ' '.join(final_caption)