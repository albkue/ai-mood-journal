import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download

# ============================================================
# 1. Configuration — must match training in BI-LSTM.py
# ============================================================
REPO_ID = "Vongvathana/BI-LSTM"
MAX_LENGTH = 35  # Same as max_len=35 used during training
THRESHOLD = 0.25  # Lower threshold catches minority emotion classes

EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# ============================================================
# 2. Download files from Hugging Face (cached after first run)
# ============================================================
print("Downloading model and tokenizer from Hugging Face (cached after first run)...")
model_path = hf_hub_download(repo_id=REPO_ID, filename="emotion_model.h5")
tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.pkl")

# ============================================================
# 3. Load the Keras model and Tokenizer
# ============================================================
print("Loading model...")
model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Model and tokenizer loaded successfully!")

# ============================================================
# 4. Prediction function (reusable in your app)
# ============================================================
def predict_goemotions(text: str, threshold: float = THRESHOLD) -> dict:
    """
    Predict emotions from text using the Bi-LSTM model.

    Args:
        text: Input text to analyze.
        threshold: Minimum probability to include an emotion (default 0.25).

    Returns:
        dict with 'emotions' (dict of emotion->score) and 'top_emotion' (str).
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    probs = model.predict(padded, verbose=0)[0]  # shape: (28,)

    detected = {
        EMOTION_COLS[i]: round(float(p), 4)
        for i, p in enumerate(probs)
        if p > threshold
    }

    # Sort by confidence descending
    detected = dict(sorted(detected.items(), key=lambda x: x[1], reverse=True))

    top_emotion = max(zip(EMOTION_COLS, probs), key=lambda x: x[1])[0] if len(probs) > 0 else "neutral"

    return {
        "text": text,
        "emotions": detected if detected else {"neutral": 1.0},
        "top_emotion": top_emotion,
        "threshold_used": threshold
    }


# ============================================================
# 5. Test it
# ============================================================
if __name__ == "__main__":
    test_texts = [
        "I am feeling incredibly happy and excited today!",
        "I'm so incredibly thankful for all your help, but I'm also terrified of what happens next!",
        "Everything feels pointless and I don't know what to do."
    ]

    for text in test_texts:
        result = predict_goemotions(text)
        print(f"\nText: {result['text']}")
        print(f"Top emotion: {result['top_emotion']}")
        print("Detected emotions:")
        for emotion, score in result['emotions'].items():
            print(f"  - {emotion}: {score*100:.2f}%")