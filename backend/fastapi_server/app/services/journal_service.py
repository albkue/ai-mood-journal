from typing import List
from sqlalchemy.orm import Session
from app import schemas
from app.models import User
from app.repositories import JournalRepository


class JournalService:
    def __init__(self):
        self.journal_repo = JournalRepository()

    def create_entry(self, db: Session, user: User, content: str) -> schemas.JournalEntry:
        entry = self.journal_repo.create_entry(db, user.id, content)
        return entry

    def get_entries(self, db: Session, user: User, skip: int = 0, limit: int = 100) -> List[schemas.JournalEntry]:
        return self.journal_repo.get_by_user(db, user.id, skip, limit)

    def get_entry(self, db: Session, user: User, entry_id: int) -> schemas.JournalEntry:
        entry = self.journal_repo.get_by_id_and_user(db, entry_id, user.id)
        if not entry:
            raise ValueError("Entry not found")
        return entry

    def update_entry(self, db: Session, user: User, entry_id: int, content: str) -> schemas.JournalEntry:
        entry = self.journal_repo.get_by_id_and_user(db, entry_id, user.id)
        if not entry:
            raise ValueError("Entry not found")
        
        # Update content and clear analysis (will be re-analyzed by ML)
        entry.content = content
        entry = self.journal_repo.clear_analysis(db, entry)
        return entry

    def delete_entry(self, db: Session, user: User, entry_id: int) -> bool:
        entry = self.journal_repo.get_by_id_and_user(db, entry_id, user.id)
        if not entry:
            raise ValueError("Entry not found")
        return self.journal_repo.delete(db, entry_id)

    def get_insights(self, db: Session, user: User) -> schemas.MoodInsight:
        entries = self.journal_repo.get_analyzed_entries(db, user.id)
        
        total_entries = len(entries)
        if total_entries == 0:
            return schemas.MoodInsight(
                total_entries=0,
                average_mood=0.0,
                mood_trend="No data",
                entries=[]
            )
        
        average_mood = sum(entry.mood_score for entry in entries) / total_entries
        
        # Simple trend calculation
        if total_entries >= 2:
            recent_mood = entries[-1].mood_score
            older_mood = entries[0].mood_score
            if recent_mood > older_mood:
                trend = "Improving"
            elif recent_mood < older_mood:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        return schemas.MoodInsight(
            total_entries=total_entries,
            average_mood=round(average_mood, 2),
            mood_trend=trend,
            entries=entries
        )
