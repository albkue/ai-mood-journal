from typing import Dict, Tuple
from enum import Enum
import random

class EmotionLabel(str, Enum):
    # GoEmotions 27 emotion labels + neutral
    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    ANGER = "anger"
    ANNOYANCE = "annoyance"
    APPROVAL = "approval"
    CARING = "caring"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    GRATITUDE = "gratitude"
    GRIEF = "grief"
    JOY = "joy"
    LOVE = "love"
    NERVOUSNESS = "nervousness"
    NEUTRAL = "neutral"
    OPTIMISM = "optimism"
    PRIDE = "pride"
    REALIZATION = "realization"
    RELIEF = "relief"
    REMORSE = "remorse"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    
    # Simplified groupings for mood score
    POSITIVE = [ADMIRATION, AMUSEMENT, APPROVAL, CARING, CURIOSITY, DESIRE, 
                EXCITEMENT, GRATITUDE, JOY, LOVE, OPTIMISM, PRIDE, REALIZATION, RELIEF]
    NEGATIVE = [ANGER, ANNOYANCE, DISAPPOINTMENT, DISAPPROVAL, DISGUST, 
                EMBARRASSMENT, FEAR, GRIEF, NERVOUSNESS, REMORSE, SADNESS]
    NEUTRAL_GROUP = [NEUTRAL, CONFUSION, SURPRISE]


class EmotionPredictor:
    """
    Emotion prediction service.
    
    This is a placeholder implementation. Replace with actual models:
    - Random Forest with TF-IDF features
    - Bi-LSTM with word embeddings
    - DistilBERT for transformer-based classification
    """
    
    # GoEmotions keyword mappings (placeholder)
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
    
    def __init__(self, model_type: str = "placeholder"):
        """
        Initialize predictor with specified model type.
        
        model_type: "random_forest", "bilstm", "distilbert", or "placeholder"
        """
        self.model_type = model_type
        self._load_model()
    
    def _load_model(self):
        """Load the ML model"""
        # TODO: Load actual models here
        # if self.model_type == "random_forest":
        #     self.model = joblib.load("models/random_forest.pkl")
        # elif self.model_type == "bilstm":
        #     self.model = load_model("models/bilstm.h5")
        # elif self.model_type == "distilbert":
        #     self.model = pipeline("text-classification", model="models/distilbert")
        pass
    
    def predict(self, text: str) -> Tuple[EmotionLabel, float]:
        """
        Predict emotion from text.
        
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        # Placeholder implementation using keyword matching
        text_lower = text.lower()
        scores = {}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[emotion] = score
        
        if max(scores.values()) == 0:
            return EmotionLabel.NEUTRAL, 0.5
        
        predicted_emotion = max(scores, key=scores.get)
        confidence = min(0.5 + (scores[predicted_emotion] * 0.1), 0.95)
        
        return predicted_emotion, confidence
    
    def predict_batch(self, texts: list) -> list:
        """Predict emotions for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def get_mood_score(self, emotion: EmotionLabel, confidence: float) -> float:
        """
        Convert GoEmotions emotion to numerical mood score (0.0 to 1.0)
        """
        # High positive emotions (0.8-1.0)
        high_positive = [
            EmotionLabel.JOY, EmotionLabel.LOVE, EmotionLabel.GRATITUDE,
            EmotionLabel.EXCITEMENT, EmotionLabel.PRIDE, EmotionLabel.RELIEF
        ]
        
        # Medium positive emotions (0.6-0.8)
        medium_positive = [
            EmotionLabel.ADMIRATION, EmotionLabel.AMUSEMENT, EmotionLabel.APPROVAL,
            EmotionLabel.CARING, EmotionLabel.CURIOSITY, EmotionLabel.DESIRE,
            EmotionLabel.OPTIMISM, EmotionLabel.REALIZATION
        ]
        
        # Neutral emotions (0.4-0.6)
        neutral = [EmotionLabel.NEUTRAL, EmotionLabel.CONFUSION, EmotionLabel.SURPRISE]
        
        # Medium negative emotions (0.2-0.4)
        medium_negative = [
            EmotionLabel.ANNOYANCE, EmotionLabel.DISAPPOINTMENT, EmotionLabel.DISAPPROVAL,
            EmotionLabel.NERVOUSNESS, EmotionLabel.REMORSE
        ]
        
        # High negative emotions (0.0-0.2)
        high_negative = [
            EmotionLabel.ANGER, EmotionLabel.DISGUST, EmotionLabel.EMBARRASSMENT,
            EmotionLabel.FEAR, EmotionLabel.GRIEF, EmotionLabel.SADNESS
        ]
        
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
        
        # Adjust by confidence
        return base_score * confidence + (0.5 * (1 - confidence))


# Singleton instance
_predictor = None

def get_predictor(model_type: str = "placeholder") -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor(model_type)
    return _predictor
