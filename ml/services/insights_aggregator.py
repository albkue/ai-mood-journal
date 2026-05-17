"""
Insights Aggregator - Computes aggregated mood insights from analyzed entries.

Responsible for:
- Daily/weekly/monthly trends
- Mood streaks
- Time-of-day effects
- Emotion & topic distributions
- Volatility and statistics

Reads from DB (pre-computed per-entry data) and caches results.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import time


class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expire_time)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expire_time = self._cache[key]
            if time.time() < expire_time:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300, ttl: int = None):
        if ttl is not None:
            ttl_seconds = ttl
        self._cache[key] = (value, time.time() + ttl_seconds)
    
    def invalidate(self, prefix: str = ""):
        if prefix:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
        else:
            self._cache.clear()


class InsightsAggregator:
    """Computes aggregated insights from analyzed journal entries."""
    
    def __init__(self):
        self.cache = SimpleCache()
    
    def get_trends(self, entries: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
        """
        Get mood trends over a period.
        
        Args:
            entries: List of analyzed entries with mood_score, sentiment_label, created_at
            days: Number of days to look back
        
        Returns:
            {
                "average_mood": float,
                "mood_trend": str,
                "mood_volatility": float,
                "best_day": str,
                "worst_day": str,
                "emotions_distribution": Dict[str, int],
                "topics_distribution": Dict[str, float],
                "period_days": int
            }
        """
        # Check cache
        cache_key = f"trends_{days}_{len(entries)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Filter to period
        cutoff = datetime.utcnow() - timedelta(days=days)
        scored_entries = [
            e for e in entries
            if e.get('mood_score') is not None
            and self._parse_date(e.get('created_at')) >= cutoff
        ]
        
        if not scored_entries:
            result = {
                "average_mood": 0.0,
                "mood_trend": "no_data",
                "mood_volatility": 0.0,
                "best_day": None,
                "worst_day": None,
                "emotions_distribution": {},
                "topics_distribution": {},
                "period_days": days
            }
            self.cache.set(cache_key, result, ttl=900)  # 15 min
            return result
        
        mood_scores = [e['mood_score'] for e in scored_entries]
        
        # Calculate statistics
        avg_mood = statistics.mean(mood_scores)
        volatility = statistics.stdev(mood_scores) if len(mood_scores) > 1 else 0.0
        
        # Find best/worst days
        best_entry = max(scored_entries, key=lambda x: x['mood_score'])
        worst_entry = min(scored_entries, key=lambda x: x['mood_score'])
        
        # Emotion distribution
        emotion_dist = defaultdict(int)
        for e in scored_entries:
            emotion = e.get('sentiment_label', 'neutral')
            emotion_dist[emotion] += 1
        
        # Topic distribution
        texts = [e.get('content', '') for e in scored_entries]
        topic_dist = self._compute_topic_distribution(scored_entries)
        
        # Calculate trend
        trend = self._calculate_trend(mood_scores)
        
        result = {
            "average_mood": round(avg_mood, 2),
            "mood_trend": trend,
            "mood_volatility": round(volatility, 2),
            "best_day": self._format_date(best_entry.get('created_at')),
            "worst_day": self._format_date(worst_entry.get('created_at')),
            "emotions_distribution": dict(emotion_dist),
            "topics_distribution": topic_dist,
            "period_days": days
        }
        
        self.cache.set(cache_key, result, ttl=900)  # 15 min
        return result
    
    def get_daily_breakdown(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get daily mood breakdown with emotions and topics per day.
        
        Returns:
            {
                "daily_insights": [
                    {
                        "date": str,
                        "entries_count": int,
                        "average_mood": float,
                        "emotions": Dict[str, int],
                        "dominant_topic": str
                    }
                ],
                "total_days": int,
                "total_entries": int
            }
        """
        cache_key = f"daily_{len(entries)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if not entries:
            result = {"daily_insights": [], "total_days": 0, "total_entries": 0}
            self.cache.set(cache_key, result, ttl=300)  # 5 min
            return result
        
        # Group by date
        daily_entries = defaultdict(list)
        for entry in entries:
            date_key = self._format_date(entry.get('created_at'))
            daily_entries[date_key].append(entry)
        
        daily_insights = []
        for date in sorted(daily_entries.keys()):
            day_entries = daily_entries[date]
            analyzed = [e for e in day_entries if e.get('mood_score') is not None]
            
            if not analyzed:
                continue
            
            avg_mood = sum(e['mood_score'] for e in analyzed) / len(analyzed)
            
            # Emotion counts
            emotion_counts = defaultdict(int)
            for e in analyzed:
                emotion_counts[e.get('sentiment_label', 'neutral')] += 1
            
            # Dominant topic
            topic_dist = self._compute_topic_distribution(analyzed)
            dominant_topic = max(topic_dist.items(), key=lambda x: x[1])[0] if topic_dist else "general"
            
            daily_insights.append({
                "date": date,
                "entries_count": len(analyzed),
                "average_mood": round(avg_mood, 2),
                "emotions": dict(emotion_counts),
                "dominant_topic": dominant_topic
            })
        
        result = {
            "daily_insights": daily_insights,
            "total_days": len(daily_insights),
            "total_entries": len(entries)
        }
        
        self.cache.set(cache_key, result, ttl=300)  # 5 min
        return result
    
    def get_streaks(self, entries: List[Dict[str, Any]], good_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate mood streaks.
        
        A "good day" has average mood >= threshold.
        A streak is consecutive good days.
        
        Returns:
            {
                "current_streak": int,
                "longest_streak": int,
                "good_days": int,
                "total_days": int,
                "good_day_percentage": float
            }
        """
        cache_key = f"streaks_{len(entries)}_{good_threshold}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if not entries:
            result = {
                "current_streak": 0,
                "longest_streak": 0,
                "good_days": 0,
                "total_days": 0,
                "good_day_percentage": 0.0
            }
            self.cache.set(cache_key, result, ttl=600)  # 10 min
            return result
        
        # Group by date and calculate daily averages
        daily_entries = defaultdict(list)
        for entry in entries:
            if entry.get('mood_score') is not None:
                date_key = self._format_date(entry.get('created_at'))
                daily_entries[date_key].append(entry['mood_score'])
        
        daily_avg = {}
        for date, scores in daily_entries.items():
            daily_avg[date] = statistics.mean(scores)
        
        # Sort by date
        sorted_dates = sorted(daily_avg.keys())
        if not sorted_dates:
            result = {
                "current_streak": 0,
                "longest_streak": 0,
                "good_days": 0,
                "total_days": 0,
                "good_day_percentage": 0.0
            }
            self.cache.set(cache_key, result, ttl=600)
            return result
        
        # Calculate streaks
        current_streak = 0
        longest_streak = 0
        temp_streak = 0
        good_days = 0
        
        for date in sorted_dates:
            if daily_avg[date] >= good_threshold:
                temp_streak += 1
                good_days += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 0
        
        # Current streak: count backwards from most recent day
        for date in reversed(sorted_dates):
            if daily_avg[date] >= good_threshold:
                current_streak += 1
            else:
                break
        
        total_days = len(sorted_dates)
        
        result = {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "good_days": good_days,
            "total_days": total_days,
            "good_day_percentage": round(good_days / total_days * 100, 1) if total_days > 0 else 0.0
        }
        
        self.cache.set(cache_key, result, ttl=600)  # 10 min
        return result
    
    def get_time_of_day_effects(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze mood patterns by time of day.
        
        Returns:
            {
                "morning": {"avg_mood": float, "count": int, "dominant_emotion": str},
                "afternoon": {"avg_mood": float, "count": int, "dominant_emotion": str},
                "evening": {"avg_mood": float, "count": int, "dominant_emotion": str},
                "night": {"avg_mood": float, "count": int, "dominant_emotion": str}
            }
        """
        cache_key = f"timeofday_{len(entries)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        periods = {
            "morning": [],      # 6-12
            "afternoon": [],    # 12-17
            "evening": [],      # 17-21
            "night": []         # 21-6
        }
        period_emotions = {
            "morning": defaultdict(int),
            "afternoon": defaultdict(int),
            "evening": defaultdict(int),
            "night": defaultdict(int)
        }
        
        for entry in entries:
            if entry.get('mood_score') is None:
                continue
            
            created_at = entry.get('created_at')
            if isinstance(created_at, datetime):
                hour = created_at.hour
            else:
                try:
                    hour = datetime.fromisoformat(str(created_at)).hour
                except:
                    continue
            
            # Determine period
            if 6 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 17:
                period = "afternoon"
            elif 17 <= hour < 21:
                period = "evening"
            else:
                period = "night"
            
            periods[period].append(entry['mood_score'])
            emotion = entry.get('sentiment_label', 'neutral')
            period_emotions[period][emotion] += 1
        
        result = {}
        for period_name, scores in periods.items():
            if scores:
                avg = statistics.mean(scores)
                dominant = max(period_emotions[period_name].items(), key=lambda x: x[1])[0]
            else:
                avg = 0.0
                dominant = "no_data"
            
            result[period_name] = {
                "avg_mood": round(avg, 2),
                "count": len(scores),
                "dominant_emotion": dominant
            }
        
        self.cache.set(cache_key, result, ttl=1800)  # 30 min
        return result
    
    def get_weekly_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze mood patterns by day of week.
        
        Returns:
            {
                "weekday_avg": float,
                "weekend_avg": float,
                "by_day": {
                    "Monday": {"avg_mood": float, "count": int},
                    ...
                },
                "best_day_of_week": str,
                "worst_day_of_week": str
            }
        """
        cache_key = f"weekly_{len(entries)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_scores = defaultdict(list)
        
        for entry in entries:
            if entry.get('mood_score') is None:
                continue
            
            created_at = entry.get('created_at')
            if isinstance(created_at, datetime):
                day_idx = created_at.weekday()
            else:
                try:
                    day_idx = datetime.fromisoformat(str(created_at)).weekday()
                except:
                    continue
            
            day_scores[day_idx].append(entry['mood_score'])
        
        by_day = {}
        weekday_scores = []
        weekend_scores = []
        
        for idx, name in enumerate(day_names):
            scores = day_scores.get(idx, [])
            if scores:
                avg = round(statistics.mean(scores), 2)
                by_day[name] = {"avg_mood": avg, "count": len(scores)}
                if idx < 5:
                    weekday_scores.extend(scores)
                else:
                    weekend_scores.extend(scores)
            else:
                by_day[name] = {"avg_mood": 0.0, "count": 0}
        
        # Best/worst day of week
        days_with_data = {k: v for k, v in by_day.items() if v["count"] > 0}
        best_day = max(days_with_data.items(), key=lambda x: x[1]["avg_mood"])[0] if days_with_data else "no_data"
        worst_day = min(days_with_data.items(), key=lambda x: x[1]["avg_mood"])[0] if days_with_data else "no_data"
        
        result = {
            "weekday_avg": round(statistics.mean(weekday_scores), 2) if weekday_scores else 0.0,
            "weekend_avg": round(statistics.mean(weekend_scores), 2) if weekend_scores else 0.0,
            "by_day": by_day,
            "best_day_of_week": best_day,
            "worst_day_of_week": worst_day
        }
        
        self.cache.set(cache_key, result, ttl=1800)  # 30 min
        return result
    
    def get_topic_emotion_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze topic-emotion correlations.
        
        Explains WHY certain emotions are predicted for specific topics:
        - Groups entries by dominant topic
        - Calculates emotion breakdown per topic
        - Identifies patterns (e.g., "Topic 19 texts always have high love scores")
        
        Returns:
            {
                "patterns": [
                    {
                        "topic": "topic_0_time",
                        "topic_label": "time / schedule",
                        "entry_count": 15,
                        "avg_mood": 0.32,
                        "dominant_emotion": "sadness",
                        "emotion_breakdown": {"sadness": 8, "anxiety": 4, "neutral": 3},
                        "explanation": "Entries about time/schedule show 53% sadness — consider stress management techniques",
                        "pattern_strength": "strong"  # strong | moderate | weak
                    }
                ],
                "strongest_pattern": "topic_0_time",
                "total_entries_analyzed": 45
            }
        """
        cache_key = f"topic_patterns_{len(entries)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if not entries:
            result = {"patterns": [], "strongest_pattern": None, "total_entries_analyzed": 0}
            self.cache.set(cache_key, result, ttl=600)
            return result
        
        # Group entries by dominant topic
        topic_entries = defaultdict(list)
        for entry in entries:
            topic = entry.get('dominant_topic', 'general')
            if topic and topic != 'general':
                topic_entries[topic].append(entry)
        
        patterns = []
        
        for topic, topic_data in topic_entries.items():
            if len(topic_data) < 2:
                continue
            
            # Calculate mood statistics
            mood_scores = [e['mood_score'] for e in topic_data if e.get('mood_score') is not None]
            avg_mood = statistics.mean(mood_scores) if mood_scores else 0.0
            
            # Emotion breakdown
            emotion_counts = defaultdict(int)
            for e in topic_data:
                emotion = e.get('sentiment_label', 'neutral')
                emotion_counts[emotion] += 1
            
            # Dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
            dominant_count = emotion_counts[dominant_emotion]
            dominant_pct = dominant_count / len(topic_data) * 100
            
            # Pattern strength
            if dominant_pct >= 60:
                pattern_strength = "strong"
            elif dominant_pct >= 40:
                pattern_strength = "moderate"
            else:
                pattern_strength = "weak"
            
            # Generate human-readable explanation
            explanation = self._generate_topic_explanation(
                topic, dominant_emotion, dominant_pct, avg_mood, len(topic_data)
            )
            
            # Clean topic label for display
            topic_label = self._clean_topic_name(topic)
            
            patterns.append({
                "topic": topic,
                "topic_label": topic_label,
                "entry_count": len(topic_data),
                "avg_mood": round(avg_mood, 2),
                "dominant_emotion": dominant_emotion,
                "emotion_breakdown": dict(sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)),
                "explanation": explanation,
                "pattern_strength": pattern_strength,
                "dominant_emotion_percentage": round(dominant_pct, 1)
            })
        
        # Sort by pattern strength and entry count
        strength_order = {"strong": 0, "moderate": 1, "weak": 2}
        patterns.sort(key=lambda x: (strength_order.get(x['pattern_strength'], 3), -x['entry_count']))
        
        strongest = patterns[0]['topic'] if patterns else None
        
        result = {
            "patterns": patterns,
            "strongest_pattern": strongest,
            "total_entries_analyzed": len(entries)
        }
        
        self.cache.set(cache_key, result, ttl=600)
        return result
    
    def get_entry_explanation(self, entry: Dict[str, Any], all_entries: List[Dict[str, Any]]) -> str:
        """
        Generate explanation for WHY a specific entry got its emotion prediction.
        
        Uses topic-emotion patterns from user's journal history.
        """
        topic = entry.get('dominant_topic', 'general')
        emotion = entry.get('sentiment_label', 'neutral')
        
        if topic == 'general' or not all_entries:
            return f"Predicted '{emotion}' based on text analysis."
        
        # Get patterns for this topic
        patterns = self.get_topic_emotion_patterns(all_entries)
        topic_pattern = next((p for p in patterns.get('patterns', []) if p['topic'] == topic), None)
        
        if topic_pattern:
            dominant = topic_pattern['dominant_emotion']
            pct = topic_pattern['dominant_emotion_percentage']
            count = topic_pattern['entry_count']
            label = topic_pattern['topic_label']
            
            if emotion == dominant:
                return (
                    f"Predicted '{emotion}' because this entry matches '{label}', "
                    f"which shows {pct}% {dominant} across {count} similar entries in your journal."
                )
            else:
                return (
                    f"Predicted '{emotion}' as an outlier — entries about '{label}' "
                    f"usually show {pct}% {dominant}. This one differs."
                )
        
        return f"Predicted '{emotion}' based on text analysis (topic: {topic})."
    
    def _generate_topic_explanation(self, topic: str, emotion: str, pct: float, 
                                     avg_mood: float, count: int) -> str:
        """Generate human-readable explanation for a topic-emotion pattern."""
        topic_label = self._clean_topic_name(topic)
        
        # Mood descriptor
        if avg_mood >= 0.7:
            mood_desc = "positive"
        elif avg_mood >= 0.4:
            mood_desc = "mixed"
        else:
            mood_desc = "negative"
        
        # Suggestion based on emotion
        suggestions = {
            "sadness": "Consider journaling about what might help improve this area.",
            "anxiety": "Try reflecting on coping strategies that have worked before.",
            "anger": "Notice if there are triggers you can identify and manage.",
            "fear": "Consider what support systems might help with these concerns.",
            "joy": "Great pattern! Notice what makes these entries positive.",
            "love": "Beautiful trend. These entries show strong connection.",
            "gratitude": "Keep nurturing this mindset — it's great for wellbeing.",
            "neutral": "Mixed responses. Explore what shifts the mood in this area."
        }
        
        suggestion = suggestions.get(emotion, "Reflect on what influences your mood here.")
        
        return (
            f"Entries about '{topic_label}' show {pct:.0f}% {emotion} "
            f"({mood_desc} overall, n={count}). {suggestion}"
        )
    
    def _clean_topic_name(self, topic: str) -> str:
        """Convert topic ID to readable label."""
        if topic.startswith('topic_'):
            # Extract descriptive part after topic_N_
            parts = topic.split('_', 2)
            if len(parts) >= 3:
                return parts[2].replace('_', ' ')
            return topic
        return topic.replace('_', ' ').title()
    
    def invalidate_cache(self, user_id: int = None):
        """Invalidate cache when new entries are added."""
        if user_id:
            self.cache.invalidate(prefix=str(user_id))
        else:
            self.cache.clear()
    
    # --- Helper methods ---
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from mood scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
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
    
    def _compute_topic_distribution(self, entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute topic distribution from entries' topics_distribution JSON field."""
        # First try: aggregate from per-entry topics_distribution (rich data)
        aggregated = defaultdict(float)
        count = 0
        
        for entry in entries:
            entry_topics = entry.get('topics_distribution')
            if entry_topics and isinstance(entry_topics, dict):
                for topic, score in entry_topics.items():
                    aggregated[topic] += float(score)
                count += 1
        
        if count > 0:
            # Average across entries
            return {k: round(v / count, 3) for k, v in sorted(aggregated.items(), key=lambda x: x[1], reverse=True)}
        
        # Fallback: use dominant_topic field
        topic_counts = defaultdict(int)
        total = 0
        
        for entry in entries:
            topic = entry.get('dominant_topic')
            if topic and topic != 'general':
                topic_counts[topic] += 1
                total += 1
        
        if total == 0:
            return {}
        
        return {k: round(v / total, 3) for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)}
    
    def _parse_date(self, date_val) -> datetime:
        """Parse date from various formats."""
        if isinstance(date_val, datetime):
            return date_val
        try:
            return datetime.fromisoformat(str(date_val))
        except:
            return datetime.utcnow()
    
    def _format_date(self, date_val) -> str:
        """Format date as YYYY-MM-DD."""
        dt = self._parse_date(date_val)
        return dt.strftime('%Y-%m-%d')


# Singleton instance
_aggregator = None

def get_insights_aggregator() -> InsightsAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = InsightsAggregator()
    return _aggregator
