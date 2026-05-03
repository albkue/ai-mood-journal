"""
Emotion Predictor - Strategy Pattern implementation.

Supports multiple emotion detection backends matching the system diagrams:
- KeywordEmotionPredictor: Keyword matching (fallback, always works)
- PretrainedBertPredictor: Pre-trained BERT on GoEmotions (use NOW)
- SklearnPredictor: Random Forest + TF-IDF (friend's .pkl model)
- KerasPredictor: Bi-LSTM (friend's .h5 model)
- TransformerPredictor: DistilBERT fine-tuned (friend's transformer model)

All implement the same EmotionPredictorBase interface, so the rest of
the system (EntryAnalyzer, InsightsAggregator) doesn't care which one is used.

Configuration via environment variables:
- EMOTION_PREDICTOR_TYPE: "keyword" | "bert" | "sklearn" | "keras" | "transformer"
- EMOTION_MODEL_PATH: Path to friend's model file
- EMOTION_VECTORIZER_PATH: Path to sklearn vectorizer (for sklearn only)
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
from enum import Enum
import os


# ============================================================
# GoEmotions Labels (shared across all implementations)
# ============================================================

class EmotionLabel(str, Enum):
    """GoEmotions 27 emotion labels + neutral."""
    # High positive
    JOY = "joy"
    LOVE = "love"
    GRATITUDE = "gratitude"
    EXCITEMENT = "excitement"
    PRIDE = "pride"
    RELIEF = "relief"
    # Medium positive
    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    APPROVAL = "approval"
    CARING = "caring"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    OPTIMISM = "optimism"
    REALIZATION = "realization"
    # Neutral
    NEUTRAL = "neutral"
    CONFUSION = "confusion"
    SURPRISE = "surprise"
    # Medium negative
    ANNOYANCE = "annoyance"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    NERVOUSNESS = "nervousness"
    REMORSE = "remorse"
    # High negative
    ANGER = "anger"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    FEAR = "fear"
    GRIEF = "grief"
    SADNESS = "sadness"

    # Groupings for mood score
    POSITIVE = ["admiration", "amusement", "approval", "caring", "curiosity", "desire",
                "excitement", "gratitude", "joy", "love", "optimism", "pride", "realization", "relief"]
    NEGATIVE = ["anger", "annoyance", "disappointment", "disapproval", "disgust",
                "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"]
    NEUTRAL_GROUP = ["neutral", "confusion", "surprise"]


# Label list for model output mapping
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise"
]


# ============================================================
# Abstract Base Class - The Interface
# ============================================================

class EmotionPredictorBase(ABC):
    """Abstract base class for all emotion predictors."""

    # Keyword mappings (shared fallback)
    EMOTION_KEYWORDS = {
        EmotionLabel.ADMIRATION: ['admire', 'respect', 'look up to', 'impressive', 'amazing work'],
        EmotionLabel.AMUSEMENT: ['funny', 'hilarious', 'laugh', 'lol', 'entertaining', 'amusing'],
        EmotionLabel.ANGER: ['angry', 'furious', 'rage', 'mad', 'hate', 'outraged', 'livid'],
        EmotionLabel.ANNOYANCE: ['annoyed', 'irritated', 'frustrated', 'bothered', 'ugh', 'argh'],
        EmotionLabel.APPROVAL: ['approve', 'agree', 'support', 'good job', 'well done', 'nice'],
        EmotionLabel.CARING: ['care', 'concern', 'worried about', 'protect', 'nurture'],
        EmotionLabel.CONFUSION: ['confused', 'puzzled', 'dont understand', 'unclear', 'what?', 'huh'],
        EmotionLabel.CURIOSITY: ['curious', 'wonder', 'interested', 'want to know', 'fascinated'],
        EmotionLabel.DESIRE: ['want', 'wish', 'hope', 'desire', 'long for', 'crave'],
        EmotionLabel.DISAPPOINTMENT: ['disappointed', 'let down', 'expected better', 'sadly'],
        EmotionLabel.DISAPPROVAL: ['disapprove', 'wrong', 'bad idea', 'shouldnt', 'disagree'],
        EmotionLabel.DISGUST: ['disgusting', 'gross', 'eww', 'repulsive', 'sickening'],
        EmotionLabel.EMBARRASSMENT: ['embarrassed', 'awkward', 'ashamed', 'cringe', 'mortified'],
        EmotionLabel.EXCITEMENT: ['excited', 'thrilled', 'cant wait', 'pumped', 'ecstatic'],
        EmotionLabel.FEAR: ['scared', 'afraid', 'terrified', 'fear', 'anxious', 'worried'],
        EmotionLabel.GRATITUDE: ['thankful', 'grateful', 'appreciate', 'thanks', 'blessed'],
        EmotionLabel.GRIEF: ['grief', 'mourning', 'loss', 'devastated', 'heartbroken'],
        EmotionLabel.JOY: ['joy', 'happy', 'elated', 'cheerful', 'delighted', 'bliss'],
        EmotionLabel.LOVE: ['love', 'adore', 'cherish', 'affection', 'romantic', 'heart'],
        EmotionLabel.NERVOUSNESS: ['nervous', 'anxious', 'tense', 'jitters', 'on edge'],
        EmotionLabel.NEUTRAL: ['okay', 'fine', 'alright', 'normal', 'standard'],
        EmotionLabel.OPTIMISM: ['optimistic', 'hopeful', 'positive', 'bright future', 'confident'],
        EmotionLabel.PRIDE: ['proud', 'accomplished', 'achievement', 'success', 'won'],
        EmotionLabel.REALIZATION: ['realize', 'understand', 'oh', 'ah', 'now i see', 'epiphany'],
        EmotionLabel.RELIEF: ['relieved', 'relax', 'calm', 'phew', 'glad its over'],
        EmotionLabel.REMORSE: ['sorry', 'regret', 'apologize', 'guilt', 'ashamed'],
        EmotionLabel.SADNESS: ['sad', 'depressed', 'unhappy', 'cry', 'tears', 'melancholy'],
        EmotionLabel.SURPRISE: ['surprised', 'shocked', 'wow', 'unexpected', 'didnt expect'],
    }

    @abstractmethod
    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """
        Predict emotion from text.

        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        pass

    @abstractmethod
    def predict_distribution(self, text: str) -> Dict[str, float]:
        """
        Get emotion distribution for text.

        Returns:
            Dictionary of {emotion_label: score}
        """
        pass

    def predict_batch(self, texts: List[str]) -> List[Tuple[EmotionLabel, float]]:
        """Predict emotions for multiple texts."""
        return [self.predict(text) for text in texts]

    def get_mood_score(self, emotion: EmotionLabel, confidence: float) -> float:
        """Convert GoEmotions emotion to numerical mood score (0.0 to 1.0)."""
        # High positive emotions (0.8-1.0)
        high_positive = [EmotionLabel.JOY, EmotionLabel.LOVE, EmotionLabel.GRATITUDE,
                         EmotionLabel.EXCITEMENT, EmotionLabel.PRIDE, EmotionLabel.RELIEF]
        # Medium positive emotions (0.6-0.8)
        medium_positive = [EmotionLabel.ADMIRATION, EmotionLabel.AMUSEMENT, EmotionLabel.APPROVAL,
                           EmotionLabel.CARING, EmotionLabel.CURIOSITY, EmotionLabel.DESIRE,
                           EmotionLabel.OPTIMISM, EmotionLabel.REALIZATION]
        # Neutral emotions (0.4-0.6)
        neutral = [EmotionLabel.NEUTRAL, EmotionLabel.CONFUSION, EmotionLabel.SURPRISE]
        # Medium negative emotions (0.2-0.4)
        medium_negative = [EmotionLabel.ANNOYANCE, EmotionLabel.DISAPPOINTMENT, EmotionLabel.DISAPPROVAL,
                           EmotionLabel.NERVOUSNESS, EmotionLabel.REMORSE]
        # High negative emotions (0.0-0.2)
        high_negative = [EmotionLabel.ANGER, EmotionLabel.DISGUST, EmotionLabel.EMBARRASSMENT,
                         EmotionLabel.FEAR, EmotionLabel.GRIEF, EmotionLabel.SADNESS]

        if emotion in high_positive:
            base_score = 0.9
        elif emotion in medium_positive:
            base_score = 0.7
        elif emotion in neutral:
            base_score = 0.5
        elif emotion in medium_negative:
            base_score = 0.3
        elif emotion in high_negative:
            base_score = 0.1
        else:
            base_score = 0.5

        return base_score * confidence + (0.5 * (1 - confidence))

    def _keyword_fallback_predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """Keyword-based fallback prediction."""
        text_lower = text.lower()
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[emotion] = score

        if max(scores.values()) == 0:
            return EmotionLabel.NEUTRAL, 0.5

        predicted = max(scores, key=scores.get)
        confidence = min(0.5 + (scores[predicted] * 0.1), 0.95)
        return predicted, confidence

    def _keyword_fallback_distribution(self, text: str) -> Dict[str, float]:
        """Keyword-based fallback distribution."""
        text_lower = text.lower()
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[emotion.value] = score

        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# 1. Keyword Emotion Predictor (Fallback)
# ============================================================

class KeywordEmotionPredictor(EmotionPredictorBase):
    """
    Keyword-based emotion prediction (fallback).
    No model loading required. Uses keyword matching against GoEmotions labels.
    Always works, but less accurate than ML models.
    """

    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        return self._keyword_fallback_predict(text)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        return self._keyword_fallback_distribution(text)


# ============================================================
# 2. Pre-trained BERT Predictor (Use NOW)
# ============================================================

class PretrainedBertPredictor(EmotionPredictorBase):
    """
    Pre-trained BERT model fine-tuned on GoEmotions.
    Model: monologg/bert-base-cased-goemotions-original
    Size: ~440MB (auto-downloads on first use)
    Accuracy: Best out-of-box for GoEmotions (28 labels)
    """

    # HuggingFace model options
    MODEL_OPTIONS = {
        "bert": "monologg/bert-base-cased-goemotions-original",
        "roberta": "SamLowe/roberta-base-go_emotions",
        "distilbert": "SamLowe/roberta-base-go_emotions",  # DistilBERT model was delisted; use RoBERTa as lightweight alternative
    }

    def __init__(self, model_variant: str = "bert"):
        self.model_variant = model_variant
        self.model_name = self.MODEL_OPTIONS.get(model_variant, self.MODEL_OPTIONS["bert"])
        self.classifier = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained GoEmotions classifier."""
        try:
            from transformers import pipeline
            print(f"Loading pre-trained GoEmotions model ({self.model_name})...")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all label scores
                device=-1  # CPU; use 0 for GPU
            )
            print(f"  Pre-trained GoEmotions model loaded successfully")
        except ImportError:
            print("  transformers not installed. Run: pip install transformers torch")
            self.classifier = None
        except Exception as e:
            print(f"  Error loading pre-trained model: {e}")
            self.classifier = None

    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """Predict using pre-trained BERT model."""
        if self.classifier is None:
            return self._keyword_fallback_predict(text)

        try:
            results = self.classifier(text[:512])  # BERT max length
            if results and results[0]:
                # Get top prediction
                top = max(results[0], key=lambda x: x['score'])
                label = top['label'].lower()

                # Map label to EmotionLabel
                try:
                    emotion = EmotionLabel(label)
                except ValueError:
                    emotion = EmotionLabel.NEUTRAL

                return emotion, round(top['score'], 3)
        except Exception as e:
            print(f"  BERT prediction error: {e}")

        return self._keyword_fallback_predict(text)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        """Get full emotion distribution from BERT."""
        if self.classifier is None:
            return self._keyword_fallback_distribution(text)

        try:
            results = self.classifier(text[:512])
            if results and results[0]:
                distribution = {}
                for item in results[0]:
                    label = item['label'].lower()
                    score = round(item['score'], 4)
                    if score > 0.01:  # Filter very low scores
                        distribution[label] = score
                return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"  BERT distribution error: {e}")

        return self._keyword_fallback_distribution(text)


# ============================================================
# 3. Sklearn Predictor (Random Forest + TF-IDF)
# ============================================================

class SklearnPredictor(EmotionPredictorBase):
    """
    Random Forest + TF-IDF emotion predictor.
    Loads friend's trained model from .pkl file.

    Expected files:
    - model_path: sklearn Random Forest model (.pkl)
    - vectorizer_path: TF-IDF vectorizer (.pkl)
    - labels_path: Label encoder (.pkl, optional)

    Training: friend trains on GoEmotions dataset using sklearn
    """

    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model_path = model_path or os.environ.get("EMOTION_MODEL_PATH", "")
        self.vectorizer_path = vectorizer_path or os.environ.get("EMOTION_VECTORIZER_PATH", "")
        self.model = None
        self.vectorizer = None
        self.label_names = GOEMOTIONS_LABELS
        self._load_model()

    def _load_model(self):
        """Load sklearn model and vectorizer."""
        import pickle

        try:
            if not self.model_path or not self.vectorizer_path:
                print("  Sklearn model/vectorizer paths not set, using keyword fallback")
                return

            print(f"Loading sklearn model from {self.model_path}...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)

            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            print(f"  Sklearn model loaded successfully")
        except ImportError:
            print("  pickle/sklearn not available")
            self.model = None
        except Exception as e:
            print(f"  Error loading sklearn model: {e}")
            self.model = None

    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """Predict using sklearn Random Forest."""
        if self.model is None or self.vectorizer is None:
            return self._keyword_fallback_predict(text)

        try:
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]
            top_idx = probabilities.argmax()
            label = self.label_names[top_idx]
            confidence = float(probabilities[top_idx])

            try:
                emotion = EmotionLabel(label)
            except ValueError:
                emotion = EmotionLabel.NEUTRAL

            return emotion, round(confidence, 3)
        except Exception as e:
            print(f"  Sklearn prediction error: {e}")
            return self._keyword_fallback_predict(text)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        """Get emotion distribution from sklearn model."""
        if self.model is None or self.vectorizer is None:
            return self._keyword_fallback_distribution(text)

        try:
            X = self.vectorizer.transform([text])
            probabilities = self.model.predict_proba(X)[0]

            distribution = {}
            for idx, score in enumerate(probabilities):
                if idx < len(self.label_names) and score > 0.01:
                    distribution[self.label_names[idx]] = round(float(score), 4)

            return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"  Sklearn distribution error: {e}")
            return self._keyword_fallback_distribution(text)


# ============================================================
# 4. Keras Predictor (Bi-LSTM)
# ============================================================

class KerasPredictor(EmotionPredictorBase):
    """
    Bi-LSTM emotion predictor.
    Loads friend's trained model from .h5 file.

    Expected files:
    - model_path: Keras Bi-LSTM model (.h5 or SavedModel format)
    - tokenizer_path: Tokenizer pickle (.pkl)

    Training: friend trains on GoEmotions dataset using TensorFlow/Keras
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None,
                 max_length: int = 128):
        self.model_path = model_path or os.environ.get("EMOTION_MODEL_PATH", "")
        self.tokenizer_path = tokenizer_path or os.environ.get("EMOTION_TOKENIZER_PATH", "")
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.label_names = GOEMOTIONS_LABELS
        self._load_model()

    def _load_model(self):
        """Load Keras Bi-LSTM model."""
        try:
            if not self.model_path:
                print("  Keras model path not set, using keyword fallback")
                return

            import tensorflow as tf
            import pickle

            print(f"Loading Keras Bi-LSTM model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)

            if self.tokenizer_path and os.path.exists(self.tokenizer_path):
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)

            print(f"  Keras Bi-LSTM model loaded successfully")
        except ImportError:
            print("  TensorFlow not installed. Run: pip install tensorflow")
            self.model = None
        except Exception as e:
            print(f"  Error loading Keras model: {e}")
            self.model = None

    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """Predict using Bi-LSTM model."""
        if self.model is None:
            return self._keyword_fallback_predict(text)

        try:
            import numpy as np

            # Tokenize
            if self.tokenizer:
                sequences = self.tokenizer.texts_to_sequences([text])
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                X = pad_sequences(sequences, maxlen=self.max_length)
            else:
                # Fallback: simple character/word level
                words = text.lower().split()[:self.max_length]
                X = np.array([[hash(w) % 10000 for w in words] + [0] * (self.max_length - len(words))])

            probabilities = self.model.predict(X, verbose=0)[0]
            top_idx = probabilities.argmax()
            label = self.label_names[top_idx]
            confidence = float(probabilities[top_idx])

            try:
                emotion = EmotionLabel(label)
            except ValueError:
                emotion = EmotionLabel.NEUTRAL

            return emotion, round(confidence, 3)
        except Exception as e:
            print(f"  Keras prediction error: {e}")
            return self._keyword_fallback_predict(text)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        """Get emotion distribution from Bi-LSTM."""
        if self.model is None:
            return self._keyword_fallback_distribution(text)

        try:
            import numpy as np

            if self.tokenizer:
                sequences = self.tokenizer.texts_to_sequences([text])
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                X = pad_sequences(sequences, maxlen=self.max_length)
            else:
                words = text.lower().split()[:self.max_length]
                X = np.array([[hash(w) % 10000 for w in words] + [0] * (self.max_length - len(words))])

            probabilities = self.model.predict(X, verbose=0)[0]

            distribution = {}
            for idx, score in enumerate(probabilities):
                if idx < len(self.label_names) and float(score) > 0.01:
                    distribution[self.label_names[idx]] = round(float(score), 4)

            return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"  Keras distribution error: {e}")
            return self._keyword_fallback_distribution(text)


# ============================================================
# 5. Transformer Predictor (DistilBERT Fine-tuned)
# ============================================================

class TransformerPredictor(EmotionPredictorBase):
    """
    Custom DistilBERT model fine-tuned on GoEmotions.
    Loads friend's fine-tuned transformer from a local directory.

    Expected: a directory containing pytorch_model.bin + config.json
    (or a HuggingFace model ID)

    Training: friend fine-tunes DistilBERT on GoEmotions dataset
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.environ.get("EMOTION_MODEL_PATH", "")
        self.classifier = None
        self._load_model()

    def _load_model(self):
        """Load fine-tuned transformer model."""
        try:
            if not self.model_path:
                print("  Transformer model path not set, using keyword fallback")
                return

            from transformers import pipeline

            print(f"Loading fine-tuned transformer from {self.model_path}...")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_path,
                top_k=None,
                device=-1
            )
            print(f"  Fine-tuned transformer loaded successfully")
        except ImportError:
            print("  transformers not installed. Run: pip install transformers torch")
            self.classifier = None
        except Exception as e:
            print(f"  Error loading transformer model: {e}")
            self.classifier = None

    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """Predict using fine-tuned transformer."""
        if self.classifier is None:
            return self._keyword_fallback_predict(text)

        try:
            results = self.classifier(text[:512])
            if results and results[0]:
                top = max(results[0], key=lambda x: x['score'])
                label = top['label'].lower()

                try:
                    emotion = EmotionLabel(label)
                except ValueError:
                    emotion = EmotionLabel.NEUTRAL

                return emotion, round(top['score'], 3)
        except Exception as e:
            print(f"  Transformer prediction error: {e}")

        return self._keyword_fallback_predict(text)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        """Get emotion distribution from fine-tuned transformer."""
        if self.classifier is None:
            return self._keyword_fallback_distribution(text)

        try:
            results = self.classifier(text[:512])
            if results and results[0]:
                distribution = {}
                for item in results[0]:
                    label = item['label'].lower()
                    score = round(item['score'], 4)
                    if score > 0.01:
                        distribution[label] = score
                return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"  Transformer distribution error: {e}")

        return self._keyword_fallback_distribution(text)


# ============================================================
# Factory Function - Environment-Configured Selection
# ============================================================

_predictor = None

def get_predictor(model_type: str = None, **kwargs) -> EmotionPredictorBase:
    """
    Factory function to get the appropriate emotion predictor.

    Selects based on:
    1. Explicit model_type parameter
    2. EMOTION_PREDICTOR_TYPE environment variable
    3. Default: "keyword"

    Args:
        model_type: "keyword" | "bert" | "sklearn" | "keras" | "transformer"
        **kwargs: Additional args passed to predictor constructor

    Returns:
        EmotionPredictorBase implementation
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    # Determine model type
    if model_type is None:
        model_type = os.environ.get("EMOTION_PREDICTOR_TYPE", "keyword").lower()

    print(f"Initializing EmotionPredictor (type={model_type})...")

    if model_type == "keyword":
        _predictor = KeywordEmotionPredictor()

    elif model_type == "bert":
        variant = kwargs.get("model_variant") or os.environ.get("BERT_MODEL_VARIANT", "bert").lower()
        _predictor = PretrainedBertPredictor(model_variant=variant)

    elif model_type == "sklearn":
        model_path = kwargs.get("model_path")
        vectorizer_path = kwargs.get("vectorizer_path")
        _predictor = SklearnPredictor(model_path=model_path, vectorizer_path=vectorizer_path)

    elif model_type == "keras":
        model_path = kwargs.get("model_path")
        tokenizer_path = kwargs.get("tokenizer_path")
        _predictor = KerasPredictor(model_path=model_path, tokenizer_path=tokenizer_path)

    elif model_type == "transformer":
        model_path = kwargs.get("model_path")
        _predictor = TransformerPredictor(model_path=model_path)

    else:
        print(f"  Unknown model type '{model_type}', falling back to keyword")
        _predictor = KeywordEmotionPredictor()

    return _predictor


def reset_predictor():
    """Reset singleton (useful for testing or switching predictors)."""
    global _predictor
    _predictor = None
