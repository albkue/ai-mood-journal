"""
Insights Service - Thin wrapper delegating to EntryAnalyzer and InsightsAggregator.

This service provides a unified interface while keeping concerns separated:
- EntryAnalyzer: per-entry analysis (emotion, topic, mood score)
- InsightsAggregator: aggregated insights (trends, streaks, patterns)
"""
from typing import List, Dict, Any

from services.entry_analyzer import get_entry_analyzer, EntryAnalyzer
from services.insights_aggregator import get_insights_aggregator, InsightsAggregator


class InsightsService:
    """Unified interface for ML insights, delegating to specialized services."""
    
    def __init__(self, topic_model_type: str = None):
        self.entry_analyzer: EntryAnalyzer = get_entry_analyzer(topic_model_type)
        self.insights_aggregator: InsightsAggregator = get_insights_aggregator()
    
    # --- Per-entry analysis (delegates to EntryAnalyzer) ---
    
    def analyze_entry(self, text: str) -> Dict[str, Any]:
        """Analyze a single journal entry."""
        return self.entry_analyzer.analyze(text)
    
    def analyze_entry_detailed(self, text: str) -> Dict[str, Any]:
        """Analyze entry with full topic distribution and mood category."""
        return self.entry_analyzer.analyze_with_details(text)
    
    # --- Aggregated insights (delegates to InsightsAggregator) ---
    
    def get_mood_trends(self, entries: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
        """Get mood trends over a period."""
        return self.insights_aggregator.get_trends(entries, days)
    
    def aggregate_daily_emotions(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get daily mood breakdown."""
        return self.insights_aggregator.get_daily_breakdown(entries)
    
    def get_streaks(self, entries: List[Dict[str, Any]], good_threshold: float = 0.5) -> Dict[str, Any]:
        """Get mood streaks."""
        return self.insights_aggregator.get_streaks(entries, good_threshold)
    
    def get_time_of_day_effects(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get mood patterns by time of day."""
        return self.insights_aggregator.get_time_of_day_effects(entries)
    
    def get_weekly_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get mood patterns by day of week."""
        return self.insights_aggregator.get_weekly_patterns(entries)
    
    def get_topic_emotion_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get topic-emotion correlation patterns."""
        return self.insights_aggregator.get_topic_emotion_patterns(entries)
    
    def get_entry_explanation(self, entry: Dict[str, Any], all_entries: List[Dict[str, Any]]) -> str:
        """Generate explanation for why an entry got its emotion prediction."""
        return self.insights_aggregator.get_entry_explanation(entry, all_entries)
    
    # --- Cache management ---
    
    def invalidate_cache(self, user_id: int = None):
        """Invalidate cached insights."""
        self.insights_aggregator.invalidate_cache(user_id)


# Singleton instance
_insights_service = None

def get_insights_service(topic_model_type: str = None) -> InsightsService:
    global _insights_service
    if _insights_service is None:
        _insights_service = InsightsService(topic_model_type)
    return _insights_service
