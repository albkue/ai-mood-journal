from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    journal_entries = relationship("JournalEntry", back_populates="owner")


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="")
    content = Column(Text)
    mood_score = Column(Float, nullable=True)
    mood_category = Column(String, nullable=True)  # great | good | okay | low | rough
    sentiment_label = Column(String, nullable=True)  # Top emotion (for quick queries)
    dominant_topic = Column(String, nullable=True)  # Top topic (for quick queries)
    emotion_distribution = Column(JSON, nullable=True)  # {"joy": 0.6, "excitement": 0.2}
    topics_distribution = Column(JSON, nullable=True)  # {"topic_7_time": 0.15, "topic_2_like": 0.12}
    analysis_status = Column(String, default="pending")  # pending | analyzed | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="journal_entries")
