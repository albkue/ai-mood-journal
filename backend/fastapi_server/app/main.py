from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import auth, journal, ml

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Mood Journal API",
    description="Backend API for AI-powered mood tracking application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(journal.router)
app.include_router(ml.router)

@app.get("/")
async def root():
    return {"message": "Welcome to AI Mood Journal API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
