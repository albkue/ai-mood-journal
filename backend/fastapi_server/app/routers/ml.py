from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app import schemas, models
from app.database import get_db
from app.auth import get_current_active_user
from app.repositories import JournalRepository

# Import ML services (adjust path as needed)
import sys
sys.path.append('/app/../ml')
from ml.services.emotion_predictor import get_predictor
from ml.services.insights_service import get_insights_service

router = APIRouter(prefix="/ml", tags=["machine_learning"])

# Initialize services
predictor = get_predictor()
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


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(
    request: AnalyzeRequest,
    current_user: models.User = Depends(get_current_active_user)
):
    """Analyze text and return emotion and topic predictions"""
    result = insights_service.analyze_entry(request.text)
    return AnalyzeResponse(**result)


@router.post("/entries/{entry_id}/analyze")
def analyze_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Analyze a specific journal entry and save results"""
    entry = journal_repo.get_by_id_and_user(db, entry_id, current_user.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    # Analyze the entry
    result = insights_service.analyze_entry(entry.content)
    
    # Update entry with analysis results
    entry = journal_repo.update_analysis(
        db, entry, 
        mood_score=result['mood_score'],
        sentiment_label=result['emotion']
    )
    
    return {
        "entry_id": entry.id,
        "emotion": result['emotion'],
        "mood_score": result['mood_score'],
        "dominant_topic": result['dominant_topic']
    }


@router.get("/insights", response_model=schemas.MoodInsight)
def get_ml_insights(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get ML-powered mood insights"""
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    
    # Convert to dict format for insights service
    entries_data = [
        {
            "id": e.id,
            "content": e.content,
            "mood_score": e.mood_score,
            "sentiment_label": e.sentiment_label,
            "created_at": e.created_at
        }
        for e in entries
    ]
    
    # Get trends
    trends = insights_service.get_mood_trends(entries_data)
    
    # Get daily aggregation
    daily = insights_service.aggregate_daily_emotions(entries_data)
    
    return schemas.MoodInsight(
        total_entries=len(entries),
        average_mood=trends['average_mood'],
        mood_trend=daily['trend'],
        entries=entries,
        emotions_distribution=trends['emotions_distribution'],
        topics_distribution=trends['topics_distribution'],
        mood_volatility=trends['mood_volatility'],
        best_day=trends['best_day'],
        worst_day=trends['worst_day']
    )


@router.post("/entries/analyze-all")
def analyze_all_entries(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Analyze all unanalyzed entries for the current user"""
    entries = db.query(models.JournalEntry).filter(
        models.JournalEntry.user_id == current_user.id,
        models.JournalEntry.mood_score.is_(None)
    ).all()
    
    analyzed_count = 0
    for entry in entries:
        result = insights_service.analyze_entry(entry.content)
        journal_repo.update_analysis(
            db, entry,
            mood_score=result['mood_score'],
            sentiment_label=result['emotion']
        )
        analyzed_count += 1
    
    return {
        "message": f"Analyzed {analyzed_count} entries",
        "analyzed_count": analyzed_count
    }
