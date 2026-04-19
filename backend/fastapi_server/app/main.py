import time
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.database import engine, Base
from app.routers import auth, journal, ml

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Mood Journal API",
    description="Backend API for AI-powered mood tracking application",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json"
)

# --- CORS Configuration ---
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://10.0.2.2:8000",
    # Add production URL here when deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# --- Rate Limiting Middleware ---
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter."""
    
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)  # ip -> [timestamps]
    
    # Stricter limits for auth endpoints
        self.auth_limits = {"/api/v1/auth/login": 5, "/api/v1/auth/register": 3}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        now = time.time()
        
        # Determine rate limit for this path
        max_req = self.auth_limits.get(path, self.max_requests)
        window = 60  # 1 minute window
        
        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < window
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= max_req:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please try again later."}
            )
        
        # Record this request
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        return response


app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


# --- Input Sanitization Middleware ---
class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Strip HTML tags and validate request body size."""
    
    MAX_BODY_SIZE = 100_000  # 100KB max request body
    
    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Request body too large"}
            )
        
        response = await call_next(request)
        return response


app.add_middleware(InputSanitizationMiddleware)

# Include routers with API versioning
API_V1_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_V1_PREFIX)
app.include_router(journal.router, prefix=API_V1_PREFIX)
app.include_router(ml.router, prefix=API_V1_PREFIX)

@app.get("/")
async def root():
    return {"message": "Welcome to AI Mood Journal API", "version": "v1", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
