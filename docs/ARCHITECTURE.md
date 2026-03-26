# System Architecture

## Overview

AI Mood Journal is a full-stack application for tracking and analyzing emotional well-being through journal entries. The system uses machine learning to automatically detect emotions and topics from text.

---

## High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flutter App   │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   (Frontend)    │◀────│   (Backend)     │◀────│   (Database)    │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   ML Pipeline   │
                        │   - LDA Topics  │
                        │   - Emotions    │
                        └─────────────────┘
```

---

## Components

### 1. Frontend (Flutter)

**Technology:** Flutter 3.x with Dart

**Architecture:** Provider State Management

**Screens:**
| Screen | Description |
|--------|-------------|
| Login | User authentication |
| Register | New user registration |
| Home | Bottom navigation, dashboard |
| Journal Entry | Create/edit journal entries |
| Entries List | View all entries |
| Insights | Mood analytics and trends |

**Key Files:**
```
lib/
├── main.dart
├── screens/
│   ├── login_screen.dart
│   ├── register_screen.dart
│   ├── home_screen.dart
│   ├── journal_entry_screen.dart
│   ├── entries_list_screen.dart
│   └── insights_screen.dart
├── providers/
│   └── auth_provider.dart
└── services/
    ├── api_service.dart
    └── auth_service.dart
```

---

### 2. Backend (FastAPI)

**Technology:** FastAPI with Python 3.12

**Architecture:** Repository Pattern

**Pattern Flow:**
```
Router → Service → Repository → Database
```

**Directory Structure:**
```
app/
├── main.py              # FastAPI app entry
├── database.py          # Database connection
├── models.py            # SQLAlchemy models
├── schemas.py           # Pydantic schemas
├── auth.py              # JWT authentication
├── repositories/
│   ├── user_repository.py
│   └── journal_repository.py
├── routers/
│   ├── auth.py
│   ├── journal.py
│   └── ml.py
└── services/
    └── journal_service.py
```

**API Endpoints:**
| Router | Base Path | Description |
|--------|-----------|-------------|
| auth | `/auth` | Authentication |
| journal | `/journal` | CRUD operations |
| ml | `/ml` | ML analysis |

---

### 3. Database (PostgreSQL)

**Technology:** PostgreSQL 15

**Tables:**

#### Users
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| email | String | Unique email |
| username | String | Unique username |
| hashed_password | String | Bcrypt hash |
| created_at | DateTime | Creation timestamp |

#### Journal Entries
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| user_id | Integer | Foreign key → users |
| title | String | Entry title |
| content | Text | Entry content |
| mood_rating | Integer | User mood (1-5) |
| mood_score | Float | ML mood (0-1) |
| sentiment_label | String | Predicted emotion |
| dominant_topic | String | LDA topic |
| created_at | DateTime | Creation timestamp |
| updated_at | DateTime | Update timestamp |

---

### 4. ML Pipeline

#### Text Preprocessing
```
Input Text → Lowercase → Remove URLs → Remove Punctuation → Tokenize → Remove Stopwords → Lemmatize
```

#### Emotion Prediction (GoEmotions)
- **Dataset:** GoEmotions (28 emotion labels)
- **Models Supported:**
  - Random Forest (sklearn)
  - Bi-LSTM (TensorFlow)
  - DistilBERT (Transformers)

**Emotion Labels:**
| Category | Emotions |
|----------|----------|
| Positive | joy, love, gratitude, excitement, pride, relief, admiration, amusement, approval, caring, curiosity, desire, optimism, realization |
| Neutral | neutral, confusion, surprise |
| Negative | anger, disgust, embarrassment, fear, grief, sadness, annoyance, disappointment, disapproval, nervousness, remorse |

#### Topic Modeling (LDA)
- **Model:** Gensim LDA
- **Topics:** 20 topics
- **Training Data:** Journal entries corpus

**Topic Examples:**
| Topic | Top Words |
|-------|-----------|
| topic_0 | good, year, point, made, old |
| topic_2 | like, lol, sound, happy, new |
| topic_7 | time, great, keep, every, hard |

---

## Data Flow

### Journal Entry Creation

```
1. User writes entry in Flutter app
2. POST /journal/ with entry data
3. Backend saves to PostgreSQL
4. ML pipeline processes content:
   a. Text preprocessing
   b. Emotion prediction → sentiment_label, mood_score
   c. Topic modeling → dominant_topic
5. Updated entry saved to database
6. Response sent to frontend
```

### Mood Insights Generation

```
1. User requests insights
2. GET /ml/insights
3. Backend fetches all user entries
4. Insights service calculates:
   - Average mood score
   - Daily mood trends
   - Emotion distribution
   - Topic distribution
   - Mood volatility
   - Best/worst days
5. Response sent to frontend
```

---

## Security

### Authentication
- **Method:** JWT Bearer Token
- **Algorithm:** HS256
- **Token Expiry:** 30 minutes (configurable)

### Password Security
- **Hashing:** Bcrypt
- **Salt:** Auto-generated per password

### API Security
- All `/journal/*` and `/ml/*` endpoints require authentication
- CORS enabled for frontend origin
- Input validation via Pydantic

---

## Deployment

### Docker Compose

```yaml
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: moodjournal
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./backend/fastapi_server
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/moodjournal
    depends_on:
      - db
    ports:
      - "8000:8000"
```

### Recommended Production Setup

- **Frontend:** Firebase Hosting / Vercel
- **Backend:** AWS ECS / Google Cloud Run
- **Database:** AWS RDS / Google Cloud SQL
- **ML Models:** AWS SageMaker / Google AI Platform
