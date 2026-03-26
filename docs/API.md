# API Documentation

## Base URL

```
Development: http://localhost:8000
Production: https://your-domain.com
```

## Authentication

All protected endpoints require JWT Bearer token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

---

## Auth Endpoints

### Register User

```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "securepassword123"
}
```

**Response (201):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2026-03-26T10:00:00"
}
```

---

### Login

```http
POST /auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### Get Current User

```http
GET /auth/me
```

**Headers:** `Authorization: Bearer <token>`

**Response (200):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2026-03-26T10:00:00"
}
```

---

## Journal Endpoints

### Create Journal Entry

```http
POST /journal/
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "title": "My Day",
  "content": "Today was a great day! I finished my project and celebrated with colleagues.",
  "mood_rating": 4
}
```

**Response (201):**
```json
{
  "id": 1,
  "title": "My Day",
  "content": "Today was a great day!...",
  "mood_rating": 4,
  "mood_score": 0.72,
  "sentiment_label": "joy",
  "dominant_topic": "topic_7_time",
  "created_at": "2026-03-26T10:00:00",
  "updated_at": "2026-03-26T10:00:00"
}
```

---

### Get All Journal Entries

```http
GET /journal/
```

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | int | 0 | Number of entries to skip |
| `limit` | int | 100 | Max entries to return |

**Response (200):**
```json
[
  {
    "id": 1,
    "title": "My Day",
    "content": "Today was a great day!...",
    "mood_rating": 4,
    "mood_score": 0.72,
    "sentiment_label": "joy",
    "dominant_topic": "topic_7_time",
    "created_at": "2026-03-26T10:00:00"
  }
]
```

---

### Get Single Entry

```http
GET /journal/{entry_id}
```

**Headers:** `Authorization: Bearer <token>`

**Response (200):**
```json
{
  "id": 1,
  "title": "My Day",
  "content": "Today was a great day!...",
  "mood_rating": 4,
  "mood_score": 0.72,
  "sentiment_label": "joy",
  "dominant_topic": "topic_7_time",
  "created_at": "2026-03-26T10:00:00"
}
```

---

### Update Entry

```http
PUT /journal/{entry_id}
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "title": "Updated Title",
  "content": "Updated content...",
  "mood_rating": 5
}
```

---

### Delete Entry

```http
DELETE /journal/{entry_id}
```

**Headers:** `Authorization: Bearer <token>`

**Response (204):** No content

---

## ML Endpoints

### Analyze Text

```http
POST /ml/analyze
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "text": "I had a wonderful day at work today!"
}
```

**Response (200):**
```json
{
  "emotion": "joy",
  "confidence": 0.85,
  "mood_score": 0.9,
  "dominant_topic": "topic_7_time",
  "topic_confidence": 0.43
}
```

---

### Get Mood Insights

```http
GET /ml/insights
```

**Headers:** `Authorization: Bearer <token>`

**Response (200):**
```json
{
  "average_mood": 0.65,
  "mood_trend": "improving",
  "mood_volatility": 0.15,
  "best_day": "2026-03-25",
  "worst_day": "2026-03-20",
  "emotions_distribution": {
    "joy": 12,
    "optimism": 8,
    "sadness": 3
  },
  "topics_distribution": {
    "topic_7_time": 0.15,
    "topic_2_like": 0.12,
    "topic_0_good": 0.10
  },
  "daily_scores": [
    {"date": "2026-03-26", "avg_mood": 0.72, "entries_count": 2}
  ]
}
```

---

### Batch Analyze Entries

```http
POST /ml/batch-analyze
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "entry_ids": [1, 2, 3]
}
```

**Response (200):**
```json
{
  "processed": 3,
  "results": [
    {"entry_id": 1, "emotion": "joy", "mood_score": 0.85},
    {"entry_id": 2, "emotion": "sadness", "mood_score": 0.25},
    {"entry_id": 3, "emotion": "neutral", "mood_score": 0.50}
  ]
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request body"
}
```

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 404 Not Found
```json
{
  "detail": "Entry not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limits

| Endpoint Type | Limit |
|--------------|-------|
| Auth | 10 requests/minute |
| Journal | 60 requests/minute |
| ML | 30 requests/minute |
