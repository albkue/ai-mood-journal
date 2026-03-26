# Setup Guide

## Prerequisites

- **Docker** & Docker Compose
- **Python 3.12+**
- **Flutter SDK 3.x**
- **Git**

---

## Quick Start (Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mood_journal.git
cd mood_journal
```

### 2. Environment Setup

Create `.env` file in `backend/fastapi_server/`:

```env
DATABASE_URL=postgresql://postgres:postgres@db:5432/moodjournal
SECRET_KEY=your-super-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. Start Services

```bash
docker-compose up -d
```

Services will be available at:
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

### 4. Stop Services

```bash
docker-compose down
```

---

## Manual Setup

### Backend (FastAPI)

#### 1. Create Virtual Environment

```bash
cd backend/fastapi_server
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Setup PostgreSQL

Create a PostgreSQL database:

```sql
CREATE DATABASE moodjournal;
```

#### 4. Configure Environment

Create `.env` file:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/moodjournal
SECRET_KEY=your-super-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

#### 5. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### Frontend (Flutter)

#### 1. Install Dependencies

```bash
cd frontend/flutter_app
flutter pub get
```

#### 2. Configure API URL

Edit `lib/config/app_config.dart`:

```dart
class AppConfig {
  static const String apiUrl = 'http://localhost:8000';
}
```

#### 3. Run App

```bash
flutter run
```

#### 4. Build for Production

```bash
# Android
flutter build apk

# iOS
flutter build ios

# Web
flutter build web
```

---

### ML Pipeline

#### 1. Create Virtual Environment

```bash
cd ml
python -m venv ml

# Windows
.\ml\Scripts\activate

# Linux/Mac
source ml/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Test LDA Model

```bash
python tests/test_lda.py
```

#### 4. Place GoEmotions Model

Put your trained emotion model in:
```
ml/models/
├── emotion_model.pkl      # Random Forest
├── emotion_model.h5       # Bi-LSTM (TensorFlow)
└── emotion_model/         # DistilBERT (transformers)
```

---

## Project Structure

```
mood_journal/
├── backend/
│   └── fastapi_server/
│       ├── app/
│       │   ├── main.py
│       │   ├── database.py
│       │   ├── models.py
│       │   ├── schemas.py
│       │   ├── auth.py
│       │   ├── repositories/
│       │   ├── routers/
│       │   └── services/
│       └── requirements.txt
├── frontend/
│   └── flutter_app/
│       ├── lib/
│       │   ├── main.dart
│       │   ├── screens/
│       │   ├── providers/
│       │   └── services/
│       └── pubspec.yaml
├── ml/
│   ├── lda_model/           # Pre-trained LDA (ignored)
│   ├── models/              # Trained models (ignored)
│   ├── services/
│   │   ├── emotion_predictor.py
│   │   ├── topic_modeler.py
│   │   └── insights_service.py
│   ├── preprocessing/
│   ├── training/
│   └── tests/
├── docs/
├── diagrams/
├── docker-compose.yml
└── README.md
```

---

## Troubleshooting

### Database Connection Error

```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart db
```

### Flutter Build Errors

```bash
flutter clean
flutter pub get
flutter run
```

### Python Virtual Environment Issues

```bash
# Recreate venv
rm -rf venv
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Find process using port
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Kill process
kill -9 <PID>
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `SECRET_KEY` | JWT secret key | Yes |
| `ALGORITHM` | JWT algorithm (default: HS256) | No |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry (default: 30) | No |
