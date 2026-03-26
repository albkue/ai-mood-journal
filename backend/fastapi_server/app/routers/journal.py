from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import schemas, models
from app.database import get_db
from app.auth import get_current_active_user
from app.services import JournalService

router = APIRouter(prefix="/journal", tags=["journal"])
journal_service = JournalService()


@router.post("/entries", response_model=schemas.JournalEntry)
def create_entry(
    entry: schemas.JournalEntryCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Create a new journal entry. Mood analysis will be added by ML pipeline."""
    return journal_service.create_entry(db, current_user, entry.content)


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
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Update a journal entry."""
    if entry_update.content is None:
        raise HTTPException(status_code=400, detail="Content is required")
    try:
        return journal_service.update_entry(db, current_user, entry_id, entry_update.content)
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
