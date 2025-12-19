from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

try:
    from src import config
except ImportError:
    import config

def define_model(vocab_size, max_length):
    # --- REVERTED: Input Shape is 4096 (VGG16) ---
    inputs1 = Input(shape=(4096,), name="image_input")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence Model (Text)
    inputs2 = Input(shape=(max_length,), name="text_input")
    # You can keep mask_zero=True here
    se1 = Embedding(vocab_size, config.EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Compile
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model