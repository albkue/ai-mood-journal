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
        ).order_by(JournalEntry.created_at.desc()).offset(skip).limit(limit).all()

    def get_by_id_and_user(self, db: Session, entry_id: int, user_id: int) -> Optional[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.id == entry_id,
            JournalEntry.user_id == user_id
        ).first()

    def get_analyzed_entries(self, db: Session, user_id: int) -> List[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.user_id == user_id,
            JournalEntry.analysis_status == "analyzed"
        ).all()

    def get_pending_entries(self, db: Session, user_id: int) -> List[JournalEntry]:
        return db.query(JournalEntry).filter(
            JournalEntry.user_id == user_id,
            JournalEntry.analysis_status == "pending"
        ).all()

    def create_entry(self, db: Session, user_id: int, title: str, content: str) -> JournalEntry:
        return self.create(db, {
            "user_id": user_id,
            "title": title,
            "content": content,
            "analysis_status": "pending"
        })

    def update_analysis(
        self, db: Session, entry: JournalEntry,
        mood_score: float,
        sentiment_label: str,
        dominant_topic: str = None,
        mood_category: str = None,
        emotion_distribution: dict = None,
        topics_distribution: dict = None
    ) -> JournalEntry:
        return self.update(db, entry, {
            "mood_score": mood_score,
            "sentiment_label": sentiment_label,
            "dominant_topic": dominant_topic,
            "mood_category": mood_category,
            "emotion_distribution": emotion_distribution,
            "topics_distribution": topics_distribution,
            "analysis_status": "analyzed"
        })

    def mark_analysis_failed(self, db: Session, entry: JournalEntry) -> JournalEntry:
        return self.update(db, entry, {
            "analysis_status": "failed"
        })

    def clear_analysis(self, db: Session, entry: JournalEntry) -> JournalEntry:
        return self.update(db, entry, {
            "mood_score": None,
            "sentiment_label": None,
            "dominant_topic": None,
            "analysis_status": "pending"
        })
