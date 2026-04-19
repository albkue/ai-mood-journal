from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from app import schemas, models
from app.database import get_db
from app.auth import get_current_active_user
from app.services import JournalService
import io
import csv
import json
from datetime import datetime

router = APIRouter(prefix="/journal", tags=["journal"])
journal_service = JournalService()


def run_ml_analysis(entry_id: int, user_id: int):
    """Background task: Run ML analysis on a journal entry"""
    from app.database import SessionLocal
    from app.repositories import JournalRepository
    import sys, os
    _ml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ml'))
    if _ml_root not in sys.path:
        sys.path.insert(0, _ml_root)
    from services.entry_analyzer import get_entry_analyzer
    from services.insights_service import get_insights_service
    
    db = SessionLocal()
    try:
        journal_repo = JournalRepository()
        analyzer = get_entry_analyzer()
        
        entry = journal_repo.get_by_id_and_user(db, entry_id, user_id)
        if not entry:
            return
        
        result = analyzer.analyze(entry.content)
        journal_repo.update_analysis(
            db, entry,
            mood_score=result['mood_score'],
            sentiment_label=result['emotion'],
            dominant_topic=result.get('dominant_topic'),
            mood_category=result.get('mood_category'),
            emotion_distribution=result.get('emotion_distribution'),
            topics_distribution=result.get('topics_distribution')
        )
        
        # Invalidate insights cache for this user
        insights = get_insights_service()
        insights.invalidate_cache(user_id)
        
    except Exception as e:
        # Mark as failed so we can retry later
        entry = db.query(models.JournalEntry).filter(models.JournalEntry.id == entry_id).first()
        if entry:
            journal_repo = JournalRepository()
            journal_repo.mark_analysis_failed(db, entry)
        print(f"ML analysis failed for entry {entry_id}: {e}")
    finally:
        db.close()


@router.post("/entries", response_model=schemas.JournalEntry)
def create_entry(
    entry: schemas.JournalEntryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Create a new journal entry. ML analysis runs in background."""
    from app.utils.sanitization import sanitize_string, sanitize_content
    
    title = sanitize_string(entry.title, max_length=200) if entry.title else ""
    content = sanitize_content(entry.content)
    
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    new_entry = journal_service.create_entry(db, current_user, title, content)
    
    # Schedule ML analysis as background task
    background_tasks.add_task(run_ml_analysis, new_entry.id, current_user.id)
    
    return new_entry


@router.get("/entries", response_model=List[schemas.JournalEntry])
def get_entries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get all journal entries for current user."""
    return journal_service.get_entries(db, current_user, skip, limit)


@router.get("/entries/{entry_id}", response_model=schemas.JournalEntry)
def get_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get a specific journal entry."""
    try:
        return journal_service.get_entry(db, current_user, entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/entries/{entry_id}", response_model=schemas.JournalEntry)
def update_entry(
    entry_id: int,
    entry_update: schemas.JournalEntryUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Update a journal entry. Re-triggers ML analysis if content changes."""
    from app.utils.sanitization import sanitize_string, sanitize_content
    
    if entry_update.content is None and entry_update.title is None:
        raise HTTPException(status_code=400, detail="Content or title is required")
    
    title = sanitize_string(entry_update.title, max_length=200) if entry_update.title else None
    content = sanitize_content(entry_update.content) if entry_update.content else None
    
    try:
        updated_entry = journal_service.update_entry(
            db, current_user, entry_id,
            content, title
        )
        # Re-trigger ML analysis if content changed
        if content is not None:
            background_tasks.add_task(run_ml_analysis, updated_entry.id, current_user.id)
        return updated_entry
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/entries/{entry_id}")
def delete_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Delete a journal entry."""
    try:
        journal_service.delete_entry(db, current_user, entry_id)
        return {"message": "Entry deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/insights", response_model=schemas.MoodInsight)
def get_insights(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get mood insights for current user."""
    return journal_service.get_insights(db, current_user)


@router.post("/entries/{entry_id}/retry-analysis")
def retry_analysis(
    entry_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Retry ML analysis for a failed entry."""
    from app.repositories import JournalRepository
    journal_repo = JournalRepository()
    entry = journal_repo.get_by_id_and_user(db, entry_id, current_user.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    if entry.analysis_status != "failed":
        raise HTTPException(status_code=400, detail="Entry is not in failed state")
    
    # Reset to pending and retry
    journal_repo.clear_analysis(db, entry)
    background_tasks.add_task(run_ml_analysis, entry_id, current_user.id)
    
    return {"message": "Analysis retry scheduled"}


@router.get("/entries/export/json")
def export_entries_json(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Export all journal entries as JSON file."""
    from app.repositories import JournalRepository
    journal_repo = JournalRepository()
    entries = journal_repo.get_by_user(db, current_user.id, limit=10000)
    
    export_data = {
        "user_id": current_user.id,
        "username": current_user.username,
        "entries": [
            {
                "id": e.id,
                "title": e.title,
                "content": e.content,
                "mood_score": e.mood_score,
                "mood_category": e.mood_category,
                "emotion": e.sentiment_label,
                "dominant_topic": e.dominant_topic,
                "emotion_distribution": e.emotion_distribution,
                "topics_distribution": e.topics_distribution,
                "created_at": str(e.created_at),
                "updated_at": str(e.updated_at)
            }
            for e in entries
        ],
        "exported_at": str(datetime.utcnow()),
        "total": len(entries)
    }
    
    return JSONResponse(
        content=export_data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=journal_entries.json"}
    )


@router.get("/entries/export/csv")
def export_entries_csv(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Export all journal entries as CSV file."""
    from app.repositories import JournalRepository
    journal_repo = JournalRepository()
    entries = journal_repo.get_by_user(db, current_user.id, limit=10000)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "title", "content", "mood_score", "mood_category", "emotion", "dominant_topic", "created_at", "updated_at"])
    
    for e in entries:
        writer.writerow([
            e.id, e.title, e.content, e.mood_score, e.mood_category,
            e.sentiment_label, e.dominant_topic, e.created_at, e.updated_at
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=journal_entries.csv"}
    )
