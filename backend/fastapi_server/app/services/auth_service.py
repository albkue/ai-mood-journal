from datetime import timedelta
from sqlalchemy.orm import Session
from app import schemas
from app.auth import (
    verify_password, get_password_hash, 
    create_access_token, create_refresh_token,
    verify_token_type, ACCESS_TOKEN_EXPIRE_MINUTES
)
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
        
        # Create access token (short-lived)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        # Create refresh token (long-lived)
        refresh_token = create_refresh_token(data={"sub": user.username})
        
        return schemas.Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )

    def refresh_access_token(self, db: Session, refresh_token: str) -> schemas.Token:
        """Get new access token using refresh token"""
        payload = verify_token_type(refresh_token, "refresh")
        if not payload:
            raise ValueError("Invalid refresh token")
        
        username = payload.get("sub")
        user = self.user_repo.get_by_username(db, username)
        if not user:
            raise ValueError("User not found")
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        # Create new refresh token (token rotation)
        new_refresh_token = create_refresh_token(data={"sub": user.username})
        
        return schemas.Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer"
        )

    def get_current_user(self, db: Session, username: str):
        return self.user_repo.get_by_username(db, username)
