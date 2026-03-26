from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import JournalEntry
from app.repositories.base_repository import BaseRepository


class JournalRepository(BaseRepository[JournalEntry]):
    def __init__(self):
        super().__init__(JournalEntry)

    def get_by_user(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.user_id == user_id
        ).offset(skip).limit(limit).all()

    def get_by_id_and_user(self, db: Session, entry_id: int, user_id: int) -> Optional[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.id == entry_id,
            JournalEntry.user_id == user_id
        ).first()

    def get_analyzed_entries(self, db: Session, user_id: int) -> List[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.user_id == user_id,
            JournalEntry.mood_score.isnot(None)
        ).all()

    def create_entry(self, db: Session, user_id: int, content: str) -> JournalEntry:
        return self.create(db, {
            "user_id": user_id,
            "content": content
        })

    def update_analysis(self, db: Session, entry: JournalEntry, mood_score: float, sentiment_label: str) -> JournalEntry:
        return self.update(db, entry, {
            "mood_score": mood_score,
            "sentiment_label": sentiment_label
        })

    def clear_analysis(self, db: Session, entry: JournalEntry) -> JournalEntry:
        return self.update(db, entry, {
            "mood_score": None,
            "sentiment_label": None
        })
