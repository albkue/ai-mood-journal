"""
Entry Analyzer - Per-entry ML analysis service.

Responsible for:
- Emotion prediction (GoEmotions)
- Topic classification
- Mood score calculation

Called asynchronously via BackgroundTasks for each new/updated entry.
"""
from typing import Dict, Any, Tuple
from services.emotion_predictor import get_predictor, EmotionLabel
from services.topic_modeler import get_topic_modeler
from services.insights_aggregator import get_insights_aggregator


class EntryAnalyzer:
    """Analyzes individual journal entries for emotion and topic."""
    
    def __init__(self, topic_model_type: str = None, emotion_model_type: str = None):
        self.predictor = get_predictor(model_type=emotion_model_type)
        self.topic_modeler = get_topic_modeler(model_type=topic_model_type)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single journal entry.
        
        Returns:
            {
                "emotion": str,              # Top emotion label
                "confidence": float,         # Top emotion confidence
                "mood_score": float,         # Overall mood (0-1)
                "mood_category": str,        # great|good|okay|low|rough
                "dominant_topic": str,       # Top topic
                "topic_confidence": float,   # Top topic confidence
                "emotion_distribution": dict, # All emotion scores
                "topics_distribution": dict   # All topic scores
            }
        """
        # Predict emotion
        emotion, confidence = self.predictor.predict(text)
        mood_score = self.predictor.get_mood_score(emotion, confidence)
        
        # Get full emotion distribution (from model)
        emotion_dist = self.predictor.predict_distribution(text)
        # Convert numpy types to native Python types for JSON serialization
        emotion_dist = {k: float(v) for k, v in emotion_dist.items()}
        
        # Extract topic with full distribution
        topic, topic_conf = self.topic_modeler.get_dominant_topic(text)
        topics_dist = self.topic_modeler.extract_topics([text])
        # Convert numpy types to native Python types for JSON serialization
        topics_dist = {k: float(v) for k, v in topics_dist.items()}
        topics_dist = dict(list(topics_dist.items())[:5])
        
        return {
            "emotion": emotion.value,
            "confidence": round(float(confidence), 2),
            "mood_score": round(float(mood_score), 2),
            "mood_category": self._categorize_mood(mood_score),
            "dominant_topic": str(topic),
            "topic_confidence": round(float(topic_conf), 2),
            "emotion_distribution": emotion_dist,
            "topics_distribution": topics_dist
        }
    
    def analyze_with_explanation(self, text: str, user_entries: list = None) -> Dict[str, Any]:
        """
        Analyze entry with human-readable explanation.
        
        Args:
            text: Journal entry text
            user_entries: All user's previous entries (for pattern context)
        
        Returns:
            Same as analyze() + 'explanation' field
        """
        result = self.analyze(text)
        
        # Generate explanation if we have historical data
        if user_entries:
            aggregator = get_insights_aggregator()
            explanation = aggregator.get_entry_explanation(
                {
                    'dominant_topic': result['dominant_topic'],
                    'sentiment_label': result['emotion']
                },
                user_entries
            )
            result['explanation'] = explanation
        else:
            result['explanation'] = f"Predicted '{result['emotion']}' based on text analysis."
        
        return result
    
    def analyze_with_details(self, text: str) -> Dict[str, Any]:
        """Same as analyze() - now includes distributions by default."""
        return self.analyze(text)
    
    def _categorize_mood(self, mood_score: float) -> str:
        """Categorize mood score into a label."""
        if mood_score >= 0.8:
            return "great"
        elif mood_score >= 0.6:
            return "good"
        elif mood_score >= 0.4:
            return "okay"
        elif mood_score >= 0.2:
            return "low"
        else:
            return "rough"


# Singleton instance
_analyzer = None

def get_entry_analyzer(topic_model_type: str = None, emotion_model_type: str = None) -> EntryAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EntryAnalyzer(topic_model_type=topic_model_type, emotion_model_type=emotion_model_type)
    return _analyzer
