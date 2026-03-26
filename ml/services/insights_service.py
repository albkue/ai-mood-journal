from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ml.services.emotion_predictor import get_predictor, EmotionLabel
from ml.services.topic_modeler import get_topic_modeler


class InsightsService:
    """Service for generating mood insights and trends"""
    
    def __init__(self, use_lda: bool = True):
        self.predictor = get_predictor()
        # Load Gensim LDA model by default
        self.topic_modeler = get_topic_modeler(
            num_topics=20,
            use_lda=use_lda,
            model_type="gensim"
        )
    
    def analyze_entry(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single journal entry.
        
        Returns:
            {
                "emotion": str,
                "confidence": float,
                "mood_score": float (0-1),
                "dominant_topic": str,
                "topic_confidence": float
            }
        """
        # Predict emotion
        emotion, confidence = self.predictor.predict(text)
        mood_score = self.predictor.get_mood_score(emotion, confidence)
        
        # Extract topic
        topic, topic_conf = self.topic_modeler.get_dominant_topic(text)
        
        return {
            "emotion": emotion.value,
            "confidence": round(confidence, 2),
            "mood_score": round(mood_score, 2),
            "dominant_topic": topic,
            "topic_confidence": round(topic_conf, 2)
        }
    
    def aggregate_daily_emotions(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate emotions by day.
        
        Args:
            entries: List of entries with 'created_at' and 'mood_score'
        
        Returns:
            {
                "daily_scores": [{"date": str, "avg_mood": float, "entries_count": int}],
                "trend": str ("improving", "declining", "stable")
            }
        """
        if not entries:
            return {"daily_scores": [], "trend": "no_data"}
        
        # Group by date
        daily_scores = defaultdict(list)
        for entry in entries:
            date = entry['created_at'].strftime('%Y-%m-%d') if isinstance(entry['created_at'], datetime) else entry['created_at'][:10]
            if entry.get('mood_score') is not None:
                daily_scores[date].append(entry['mood_score'])
        
        # Calculate daily averages
        daily_avg = []
        for date in sorted(daily_scores.keys()):
            scores = daily_scores[date]
            daily_avg.append({
                "date": date,
                "avg_mood": round(statistics.mean(scores), 2),
                "entries_count": len(scores)
            })
        
        # Determine trend
        trend = self._calculate_trend([d['avg_mood'] for d in daily_avg])
        
        return {
            "daily_scores": daily_avg,
            "trend": trend
        }
    
    def get_mood_trends(self, entries: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
        """
        Get mood trends over time.
        
        Returns:
            {
                "average_mood": float,
                "mood_volatility": float,
                "best_day": str,
                "worst_day": str,
                "emotions_distribution": Dict[str, int],
                "topics_distribution": Dict[str, float]
            }
        """
        if not entries:
            return {
                "average_mood": 0.0,
                "mood_volatility": 0.0,
                "best_day": None,
                "worst_day": None,
                "emotions_distribution": {},
                "topics_distribution": {}
            }
        
        # Filter entries with mood scores
        scored_entries = [e for e in entries if e.get('mood_score') is not None]
        if not scored_entries:
            return {
                "average_mood": 0.0,
                "mood_volatility": 0.0,
                "best_day": None,
                "worst_day": None,
                "emotions_distribution": {},
                "topics_distribution": {}
            }
        
        mood_scores = [e['mood_score'] for e in scored_entries]
        
        # Calculate statistics
        avg_mood = statistics.mean(mood_scores)
        volatility = statistics.stdev(mood_scores) if len(mood_scores) > 1 else 0.0
        
        # Find best/worst days
        best_entry = max(scored_entries, key=lambda x: x['mood_score'])
        worst_entry = min(scored_entries, key=lambda x: x['mood_score'])
        
        # Emotion distribution
        emotions = [e.get('sentiment_label', 'neutral') for e in entries]
        emotion_dist = {}
        for emotion in emotions:
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # Topic distribution
        texts = [e.get('content', '') for e in entries]
        topic_dist = self.topic_modeler.extract_topics(texts)
        
        return {
            "average_mood": round(avg_mood, 2),
            "mood_volatility": round(volatility, 2),
            "best_day": best_entry['created_at'].strftime('%Y-%m-%d') if isinstance(best_entry['created_at'], datetime) else best_entry['created_at'][:10],
            "worst_day": worst_entry['created_at'].strftime('%Y-%m-%d') if isinstance(worst_entry['created_at'], datetime) else worst_entry['created_at'][:10],
            "emotions_distribution": emotion_dist,
            "topics_distribution": topic_dist
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from a list of scores"""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid = len(scores) // 2
        first_half = statistics.mean(scores[:mid]) if mid > 0 else scores[0]
        second_half = statistics.mean(scores[mid:]) if mid < len(scores) else scores[-1]
        
        diff = second_half - first_half
        threshold = 0.1
        
        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"


# Singleton instance
_insights_service = None

def get_insights_service() -> InsightsService:
    global _insights_service
    if _insights_service is None:
        _insights_service = InsightsService()
    return _insights_service
