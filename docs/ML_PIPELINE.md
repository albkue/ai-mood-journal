# ML Pipeline Documentation

## Overview

The ML pipeline processes journal entries to extract emotional insights and discover topics. It combines emotion classification (GoEmotions) with topic modeling (LDA).

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │    Text     │    │    Emotion      │    │     Topic       │  │
│  │ Preprocessing│───▶│   Prediction   │───▶│    Modeling     │  │
│  └─────────────┘    └─────────────────┘    └─────────────────┘  │
│         │                   │                      │            │
│         │                   ▼                      ▼            │
│         │           ┌─────────────────────────────────┐        │
│         │           │       Insights Service          │        │
│         └──────────▶│  - Daily Aggregation            │        │
│                     │  - Trend Calculation            │        │
│                     │  - Statistics                   │        │
│                     └─────────────────────────────────┘        │
│                                   │                             │
└───────────────────────────────────┼─────────────────────────────┘
                                    ▼
                            ┌─────────────┐
                            │   API       │
                            │ Response    │
                            └─────────────┘
```

---

## Components

### 1. Text Preprocessor

**File:** `ml/preprocessing/text_preprocessor.py`

**Pipeline:**
1. Lowercase conversion
2. URL removal
3. Punctuation removal
4. Tokenization
5. Stopword removal
6. Lemmatization

**Example:**
```python
from ml.preprocessing.text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
processed = preprocessor.preprocess("I had a GREAT day at work!!!")
# Output: "great day work"
```

---

### 2. Emotion Predictor

**File:** `ml/services/emotion_predictor.py`

**Dataset:** GoEmotions (Reddit comments, 28 emotion labels)

**Supported Models:**

| Model | File Format | Library |
|-------|-------------|---------|
| Random Forest | `.pkl` | scikit-learn |
| Bi-LSTM | `.h5` | TensorFlow |
| DistilBERT | folder | Transformers |

**Emotion Labels (28):**
```python
class EmotionLabel(str, Enum):
    # High Positive (mood_score: 0.9)
    JOY = "joy"
    LOVE = "love"
    GRATITUDE = "gratitude"
    EXCITEMENT = "excitement"
    PRIDE = "pride"
    RELIEF = "relief"
    
    # Medium Positive (mood_score: 0.7)
    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    APPROVAL = "approval"
    CARING = "caring"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    OPTIMISM = "optimism"
    REALIZATION = "realization"
    
    # Neutral (mood_score: 0.5)
    NEUTRAL = "neutral"
    CONFUSION = "confusion"
    SURPRISE = "surprise"
    
    # Medium Negative (mood_score: 0.3)
    ANNOYANCE = "annoyance"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    NERVOUSNESS = "nervousness"
    REMORSE = "remorse"
    
    # High Negative (mood_score: 0.1)
    ANGER = "anger"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    FEAR = "fear"
    GRIEF = "grief"
    SADNESS = "sadness"
```

**Usage:**
```python
from ml.services.emotion_predictor import get_predictor

predictor = get_predictor()
emotion, confidence = predictor.predict("I'm so happy today!")
mood_score = predictor.get_mood_score(emotion, confidence)

print(f"Emotion: {emotion.value}")  # "joy"
print(f"Confidence: {confidence}")   # 0.85
print(f"Mood Score: {mood_score}")   # 0.9
```

**Placeholder Implementation:**
Currently uses keyword-based scoring. Replace with trained model:

```python
# In emotion_predictor.py, update _load_model():

def _load_model(self):
    if self.model_type == "random_forest":
        import joblib
        self.model = joblib.load("ml/models/emotion_model.pkl")
    
    elif self.model_type == "bilstm":
        from tensorflow.keras.models import load_model
        self.model = load_model("ml/models/emotion_model.h5")
    
    elif self.model_type == "distilbert":
        from transformers import pipeline
        self.model = pipeline("text-classification", 
                            model="ml/models/emotion_model")
```

---

### 3. Topic Modeler (LDA)

**File:** `ml/services/topic_modeler.py`

**Model:** Gensim LDA with 20 topics

**Location:** `ml/lda_model/`

```
ml/lda_model/
├── lda_20topics.model           # Main model
├── lda_20topics.model.expElogbeta.npy
├── lda_20topics.model.id2word
├── lda_20topics.model.state
├── dictionary.dict              # Vocabulary
├── corpus.pkl                   # Training corpus
└── lda_visualization.html       # pyLDAvis output
```

**Usage:**
```python
from ml.services.topic_modeler import get_topic_modeler

modeler = get_topic_modeler(use_lda=True, model_type="gensim")

# Single entry
topic, score = modeler.get_dominant_topic("Great day at work!")
# Output: ("topic_7_time", 0.43)

# Multiple entries
topics = modeler.extract_topics([
    "Work was productive today",
    "Had fun with friends this weekend"
])

# Get topic keywords
words = modeler.get_topic_keywords("topic_7_time")
# Output: ["time", "great", "keep", "every", "hard"]
```

**20 Topics Overview:**
| Topic ID | Top Words |
|----------|-----------|
| 0 | good, year, point, made, old |
| 1 | thank, right, something, try, ill |
| 2 | like, lol, sound, happy, new |
| 3 | look, like, hate, damn, shes |
| 4 | problem, interesting, exactly, took, boy |
| 5 | youre, still, start, glad, literally |
| 6 | sorry, enough, suck, saying, really |
| 7 | time, great, keep, every, hard |
| 8 | much, mean, play, fun, away |
| 9 | name, bad, like, feel, he |
| 10 | well, yes, yes, hes, things |
| 11 | day, today, going, nice, weekend |
| 12 | wait, soon, oh, really, nice |
| 13 | wish, hope, better, could, would |
| 14 | really, pretty, sure, oh, nice |
| 15 | need, help, sure, someone, try |
| 16 | nice, sounds, sounds, cool, awesome |
| 17 | know, dont, think, didnt, said |
| 18 | thats, sure, true, definitely, yeah |
| 19 | hey, hi, hello, oh, thanks |

---

### 4. Insights Service

**File:** `ml/services/insights_service.py`

**Responsibilities:**
- Aggregate emotions by day
- Calculate mood trends
- Compute statistics

**Usage:**
```python
from ml.services.insights_service import get_insights_service

service = get_insights_service()

# Analyze single entry
result = service.analyze_entry("I had a wonderful day!")
# Output: {
#   "emotion": "joy",
#   "confidence": 0.85,
#   "mood_score": 0.9,
#   "dominant_topic": "topic_7_time",
#   "topic_confidence": 0.43
# }

# Get daily aggregation
daily = service.aggregate_daily_emotions(entries)
# Output: {
#   "daily_scores": [...],
#   "trend": "improving" | "declining" | "stable"
# }

# Get full insights
insights = service.get_mood_trends(entries)
# Output: {
#   "average_mood": 0.65,
#   "mood_volatility": 0.15,
#   "best_day": "2026-03-25",
#   "worst_day": "2026-03-20",
#   "emotions_distribution": {...},
#   "topics_distribution": {...}
# }
```

---

## Integration

### API Endpoint

**File:** `backend/fastapi_server/app/routers/ml.py`

```python
@router.post("/analyze")
def analyze_text(request: AnalyzeRequest, current_user = Depends(get_current_user)):
    result = insights_service.analyze_entry(request.text)
    return AnalyzeResponse(**result)

@router.get("/insights")
def get_insights(current_user = Depends(get_current_user), db = Depends(get_db)):
    entries = journal_repo.get_entries(db, current_user.id)
    return insights_service.get_mood_trends(entries)
```

---

## Training

### LDA Training (Optional)

```bash
cd ml
python training/train_lda.py --sample    # Sample data
python training/train_lda.py --from-db   # From database
```

### Emotion Model Training

1. Download GoEmotions dataset
2. Preprocess text
3. Train model (Random Forest / Bi-LSTM / DistilBERT)
4. Save to `ml/models/emotion_model.*`
5. Update `emotion_predictor.py` to load model

---

## Testing

```bash
cd ml
.\ml\Scripts\activate
python tests/test_lda.py
```

---

## Requirements

```
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
transformers>=4.35.0
torch>=2.1.0
nltk>=3.8.1
spacy>=3.7.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
gensim>=4.4.0
```

Install:
```bash
pip install -r requirements.txt
```
