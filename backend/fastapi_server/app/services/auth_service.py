from datetime import timedelta
from sqlalchemy.orm import Session
from app import schemas
from app.auth import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.repositories import UserRepository


class AuthService:
    def __init__(self):
        self.user_repo = UserRepository()

    def register(self, db: Session, user_data: schemas.UserCreate) -> schemas.User:
        # Check if email exists
        if self.user_repo.get_by_email(db, user_data.email):
            raise ValueError("Email already registered")
        
        # Check if username exists
        if self.user_repo.get_by_username(db, user_data.username):
            raise ValueError("Username already taken")
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        user = self.user_repo.create_user(
            db,
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password
        )
        return user

    def authenticate(self, db: Session, username: str, password: str) -> schemas.Token:
        user = self.user_repo.get_by_username(db, username)
        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Incorrect username or password")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return schemas.Token(access_token=access_token, token_type="bearer")

    def get_current_user(self, db: Session, username: str):
        return self.user_repo.get_by_username(db, username)
