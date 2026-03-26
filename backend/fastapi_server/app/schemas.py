from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict


class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class JournalEntryBase(BaseModel):
    content: str


class JournalEntryCreate(JournalEntryBase):
    pass


class JournalEntryUpdate(BaseModel):
    content: Optional[str] = None


class JournalEntry(JournalEntryBase):
    id: int
    user_id: int
    mood_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MoodInsight(BaseModel):
    total_entries: int
    average_mood: float
    mood_trend: str
    entries: List[JournalEntry]
    emotions_distribution: Optional[Dict[str, int]] = None
    topics_distribution: Optional[Dict[str, float]] = None
    mood_volatility: Optional[float] = None
    best_day: Optional[str] = None
    worst_day: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
