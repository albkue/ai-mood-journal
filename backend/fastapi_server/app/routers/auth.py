from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app import schemas
from app.database import get_db
from app.services import AuthService

router = APIRouter(prefix="/auth", tags=["authentication"])
auth_service = AuthService()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        return auth_service.register(db, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=schemas.Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    try:
        return auth_service.authenticate(db, login_data.username, login_data.password)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=schemas.Token)
async def refresh_token(
    refresh_data: RefreshRequest,
    db: Session = Depends(get_db)
):
    """Get new access token using refresh token"""
    try:
        return auth_service.refresh_access_token(db, refresh_data.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
