from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app import schemas, models
from app.database import get_db
from app.auth import get_current_active_user
from app.repositories import JournalRepository

# Import ML services
import sys
import os
# Add ML root directory to Python path
_ml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ml'))
if _ml_root not in sys.path:
    sys.path.insert(0, _ml_root)
from services.insights_service import get_insights_service

router = APIRouter(prefix="/ml", tags=["machine_learning"])

# Initialize services
insights_service = get_insights_service()
journal_repo = JournalRepository()


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    emotion: str
    confidence: float
    mood_score: float
    dominant_topic: str
    topic_confidence: float


def _entries_to_dicts(entries) -> List[dict]:
    """Convert SQLAlchemy entries to dicts for insights service."""
    return [
        {
            "id": e.id,
            "content": e.content,
            "mood_score": e.mood_score,
            "mood_category": e.mood_category,
            "sentiment_label": e.sentiment_label,
            "dominant_topic": e.dominant_topic,
            "emotion_distribution": e.emotion_distribution,
            "topics_distribution": e.topics_distribution,
            "created_at": e.created_at
        }
        for e in entries
    ]


# --- Per-entry analysis endpoints ---

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(
    request: AnalyzeRequest,
    current_user: models.User = Depends(get_current_active_user)
):
    """Analyze text and return emotion and topic predictions (no DB save)"""
    result = insights_service.analyze_entry(request.text)
    return AnalyzeResponse(**result)


@router.post("/entries/{entry_id}/analyze")
def analyze_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Manually trigger analysis for a specific entry"""
    entry = journal_repo.get_by_id_and_user(db, entry_id, current_user.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    result = insights_service.analyze_entry(entry.content)
    entry = journal_repo.update_analysis(
        db, entry,
        mood_score=result['mood_score'],
        sentiment_label=result['emotion'],
        dominant_topic=result.get('dominant_topic'),
        mood_category=result.get('mood_category'),
        emotion_distribution=result.get('emotion_distribution'),
        topics_distribution=result.get('topics_distribution')
    )
    
    # Invalidate cache
    insights_service.invalidate_cache(current_user.id)
    
    return {
        "entry_id": entry.id,
        "emotion": result['emotion'],
        "mood_score": result['mood_score'],
        "mood_category": result.get('mood_category'),
        "dominant_topic": result.get('dominant_topic'),
        "emotion_distribution": result.get('emotion_distribution'),
        "topics_distribution": result.get('topics_distribution'),
        "analysis_status": entry.analysis_status
    }


@router.post("/entries/analyze-all")
def analyze_all_entries(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Analyze all pending/failed entries for the current user"""
    entries = db.query(models.JournalEntry).filter(
        models.JournalEntry.user_id == current_user.id,
        models.JournalEntry.analysis_status.in_(["pending", "failed"])
    ).all()
    
    analyzed_count = 0
    failed_count = 0
    for entry in entries:
        try:
            result = insights_service.analyze_entry(entry.content)
            journal_repo.update_analysis(
                db, entry,
                mood_score=result['mood_score'],
                sentiment_label=result['emotion'],
                dominant_topic=result.get('dominant_topic'),
                mood_category=result.get('mood_category'),
                emotion_distribution=result.get('emotion_distribution'),
                topics_distribution=result.get('topics_distribution')
            )
            analyzed_count += 1
        except Exception:
            journal_repo.mark_analysis_failed(db, entry)
            failed_count += 1
    
    # Invalidate cache
    insights_service.invalidate_cache(current_user.id)
    
    return {
        "message": f"Analyzed {analyzed_count} entries, {failed_count} failed",
        "analyzed_count": analyzed_count,
        "failed_count": failed_count
    }


# --- Aggregated insights endpoints ---

@router.get("/insights")
def get_ml_insights(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get ML-powered mood insights for a period (default: 30 days)"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    entries_data = _entries_to_dicts(entries)
    
    # Get trends from aggregator
    trends = insights_service.get_mood_trends(entries_data, days)
    
    return {
        "total_entries": len(entries),
        "average_mood": trends['average_mood'],
        "mood_trend": trends['mood_trend'],
        "mood_volatility": trends['mood_volatility'],
        "best_day": trends['best_day'],
        "worst_day": trends['worst_day'],
        "emotions_distribution": trends['emotions_distribution'],
        "topics_distribution": trends['topics_distribution'],
        "period_days": trends['period_days']
    }


@router.get("/daily-insights")
def get_daily_insights(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get daily mood breakdown with emotions and topics per day"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    entries_data = _entries_to_dicts(entries)
    
    return insights_service.aggregate_daily_emotions(entries_data)


@router.get("/streaks")
def get_streaks(
    good_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get mood streaks (consecutive good days)"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    entries_data = _entries_to_dicts(entries)
    
    return insights_service.get_streaks(entries_data, good_threshold)


@router.get("/time-of-day")
def get_time_of_day(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get mood patterns by time of day (morning/afternoon/evening/night)"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    entries_data = _entries_to_dicts(entries)
    
    return insights_service.get_time_of_day_effects(entries_data)


@router.get("/weekly-patterns")
def get_weekly_patterns(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get mood patterns by day of week (weekday vs weekend comparison)"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    entries_data = _entries_to_dicts(entries)
    
    return insights_service.get_weekly_patterns(entries_data)
