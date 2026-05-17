import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import pickle
# ==========================================
# 1. LOAD AND PREPARE THE DATASET
# ==========================================
df = pd.read_csv("GoEmotions.csv")
df = df[df['example_very_unclear'] == False]
texts = df["text"].astype(str)

emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

labels = df[emotion_cols].values 

# ==========================================
# 2. TRAIN / TEST SPLIT & TOKENIZATION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

max_words = 10000 
max_len = 35  # Reduced from 100! The longest sentence in the dataset is 33 words.

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')

# ==========================================
# 3. BUILD THE ENHANCED BI-LSTM MODEL
# ==========================================
model = Sequential([
    Embedding(max_words, 100, input_length=max_len), 
    SpatialDropout1D(0.3), 
    
    # Added recurrent_dropout to prevent LSTM overfitting
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    
    GlobalMaxPooling1D(),  
    
    Dense(64, activation='relu'),
    Dropout(0.4),
    
    Dense(28, activation='sigmoid') 
])

# Use explicit multi-label metrics to get an accurate picture of performance!
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=METRICS
)

# ==========================================
# 4. TRAIN WITH CALLBACKS
# ==========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5),
    CSVLogger('training_history.csv', append=True)
]

# Note: Removed Keras `class_weight` as it breaks multi-label training. 
history = model.fit(
    X_train_pad, 
    y_train,
    epochs=12, 
    batch_size=128, # Larger batch size speeds up training and stabilizes gradients
    validation_data=(X_test_pad, y_test),
    callbacks=callbacks
)

# ==========================================
# 4.5. SAVE THE MODEL AND TOKENIZER
# ==========================================
print("\nSaving model and tokenizer...")
model.save('emotion_model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved to 'emotion_model.h5' and 'tokenizer.pkl'")

# ==========================================
# 5. PREDICTION WITH CUSTOM THRESHOLD
# ==========================================
def predict_goemotions(text, threshold=0.25):
    """
    Using a threshold lower than 0.5 helps capture minority classes 
    (like grief or pride) that struggle to reach high confidence.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    preds = model.predict(padded)[0]
    
    print(f"\nText: '{text}'")
    print(f"\n Detected Emotions (Threshold > {threshold}):")
    
    detected = False
    for i, prob in enumerate(preds):
        if prob > threshold:
            print(f"- {emotion_cols[i]}: {prob*100:.2f}%")
            detected = True
            
    if not detected:
        print("- neutral (Model was not highly confident in any specific emotion)")

predict_goemotions("I'm so incredibly thankful for all your help, but I'm also terrified of what happens next!")