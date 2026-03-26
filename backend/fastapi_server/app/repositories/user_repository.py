from typing import Optional
from sqlalchemy.orm import Session
from app.models import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self):
        super().__init__(User)

    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    def create_user(self, db: Session, email: str, username: str, hashed_password: str) -> User:
        return self.create(db, {
            "email": email,
            "username": username,
            "hashed_password": hashed_password
        })
