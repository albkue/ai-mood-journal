from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from app import schemas, models
from app.database import get_db
from app.auth import get_current_active_user
from app.services import AuthService

router = APIRouter(prefix="/auth", tags=["authentication"])
auth_service = AuthService()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    from app.utils.sanitization import sanitize_string, validate_email
    
    # Sanitize inputs
    user.username = sanitize_string(user.username, max_length=50)
    if not validate_email(user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    
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


@router.get("/export-data")
def export_user_data(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Export all user data as JSON (GDPR right to data portability).
    Returns user profile + all journal entries.
    """
    from app.repositories import JournalRepository
    journal_repo = JournalRepository()
    entries = journal_repo.get_by_user(db, current_user.id, limit=10000)
    
    export = {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.username,
            "created_at": str(current_user.created_at)
        },
        "journal_entries": [
            {
                "id": e.id,
                "title": e.title,
                "content": e.content,
                "mood_score": e.mood_score,
                "mood_category": e.mood_category,
                "sentiment_label": e.sentiment_label,
                "dominant_topic": e.dominant_topic,
                "emotion_distribution": e.emotion_distribution,
                "topics_distribution": e.topics_distribution,
                "analysis_status": e.analysis_status,
                "created_at": str(e.created_at),
                "updated_at": str(e.updated_at)
            }
            for e in entries
        ],
        "exported_at": str(datetime.utcnow()),
        "total_entries": len(entries)
    }
    
    return JSONResponse(
        content=export,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=mood_journal_export.json"}
    )


class DeleteAccountRequest(BaseModel):
    password: str
    confirm: str  # Must type "DELETE" to confirm


@router.delete("/delete-account")
def delete_account(
    request: DeleteAccountRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Permanently delete user account and all associated data.
    Requires password confirmation and typing "DELETE" to confirm.
    """
    from app.auth import verify_password
    
    # Verify password
    if not verify_password(request.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )
    
    # Verify confirmation
    if request.confirm != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Type "DELETE" to confirm account deletion'
        )
    
    # Delete all journal entries
    from app.models import JournalEntry
    db.query(JournalEntry).filter(
        JournalEntry.user_id == current_user.id
    ).delete()
    
    # Delete user
    db.delete(current_user)
    db.commit()
    
    return {"message": "Account and all data permanently deleted"}
