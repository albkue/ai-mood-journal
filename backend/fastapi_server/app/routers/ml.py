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


@router.get("/daily-insights")
def get_daily_insights(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Get daily insights with:
    - Individual entry emotions
    - Daily LDA topics (from concatenated daily texts)
    - Combined visualization data
    
    Workflow:
    1. Group entries by day
    2. Predict emotions for each entry
    3. Concatenate daily texts
    4. Run LDA on daily corpora
    5. Return combined daily emotions + topics
    """
    from collections import defaultdict
    from datetime import datetime
    
    # Get all entries for user
    entries = journal_repo.get_analyzed_entries(db, current_user.id)
    if not entries:
        return {"daily_insights": []}
    
    # Group entries by date
    daily_entries = defaultdict(list)
    for entry in entries:
        date_key = entry.created_at.strftime('%Y-%m-%d') if isinstance(entry.created_at, datetime) else str(entry.created_at)[:10]
        daily_entries[date_key].append(entry)
    
    # Process each day
    daily_insights = []
    
    for date in sorted(daily_entries.keys()):
        day_entries = daily_entries[date]
        
        # Step 1: Individual entry emotions
        entry_emotions = []
        for entry in day_entries:
            # Analyze if not already analyzed
            if entry.mood_score is None:
                result = insights_service.analyze_entry(entry.content)
                entry.mood_score = result['mood_score']
                entry.sentiment_label = result['emotion']
                db.commit()
            
            entry_emotions.append({
                "entry_id": entry.id,
                "title": entry.title,
                "emotion": entry.sentiment_label or "neutral",
                "mood_score": entry.mood_score or 0.5,
                "dominant_topic": entry.dominant_topic or "general"
            })
        
        # Step 2: Calculate daily emotion aggregation
        avg_mood = sum(e['mood_score'] for e in entry_emotions) / len(entry_emotions)
        emotion_counts = defaultdict(int)
        for e in entry_emotions:
            emotion_counts[e['emotion']] += 1
        
        # Step 3: Concatenate daily texts
        daily_text = " ".join([e.content for e in day_entries])
        
        # Step 4: Run LDA on daily corpus (single document)
        daily_topics = insights_service.topic_modeler.extract_topics([daily_text])
        dominant_daily_topic = max(daily_topics.items(), key=lambda x: x[1]) if daily_topics else ("general", 1.0)
        
        # Step 5: Compile daily insight
        daily_insights.append({
            "date": date,
            "entries_count": len(day_entries),
            "average_mood": round(avg_mood, 2),
            "emotions": dict(emotion_counts),
            "entry_details": entry_emotions,
            "daily_topics": dict(list(daily_topics.items())[:5]),
            "dominant_daily_topic": {
                "topic": dominant_daily_topic[0],
                "score": round(dominant_daily_topic[1], 4)
            },
            "daily_text_preview": daily_text[:200] + "..." if len(daily_text) > 200 else daily_text
        })
    
    return {
        "daily_insights": daily_insights,
        "total_days": len(daily_insights),
        "total_entries": len(entries)
    }
