# 📘 AI Mood Journal - Technical Codebase Documentation

This document provides a comprehensive walkthrough of the AI Mood Journal codebase. It explains exactly how the models are retrieved, how they operate together, and how they are integrated into the FastAPI backend and Flutter frontend.

---

## 📂 Codebase Directory Overview

Here are the key directories and files that drive this application:

```
ai-mood-journal/
├── backend/
│   └── fastapi_server/
│       ├── app/
│       │   ├── routers/
│       │   │   ├── auth.py          # User authentication endpoints (Sign Up / Log In)
│       │   │   ├── journal.py       # Handles saving/getting journal entries in DB
│       │   │   └── ml.py            # Machine Learning configurations & predictions
│       │   └── main.py              # Entry point of the FastAPI application
│       └── Dockerfile               # Builds the FastAPI + ML environment
├── ml/
│   ├── Bi-LSTM/
│   │   └── load_model.py            # Code that downloads & loads our Bi-LSTM model
│   ├── lda_model/                   # Stores our offline Gensim LDA dictionary & weights
│   └── services/
│       ├── entry_analyzer.py        # CORE PIPELINE: Where Bi-LSTM & LDA work together!
│       ├── insights_aggregator.py   # Aggregates daily trends, streaks, & weekly patterns
│       └── insights_service.py      # Unified wrapper for all ML insights services
├── frontend/
│   └── flutter_app/
│       └── lib/
│           ├── screens/
│           │   ├── journal_entry_screen.dart # Form where user types journal entry
│           │   └── insights_screen.dart      # Dashboard showing mood chart, streaks, & trends
│           └── services/
│               └── api_service.dart          # Directs HTTP traffic to the FastAPI server
└── docker-compose.yml               # Orchestrates the backend API and PostgreSQL DB
```

---

## 📡 1. How the Models are Retrieved & Loaded

Both models are dynamically loaded upon backend startup inside the Docker environment.

### 🎭 Bi-LSTM (Deep Learning Emotion Model)
*   **Source:** Hosted on Hugging Face at [Vongvathana/BI-LSTM](https://huggingface.co/Vongvathana/BI-LSTM).
*   **Implementation (`ml/Bi-LSTM/load_model.py`):**
    *   The `huggingface_hub` package retrieves the required files:
        ```python
        model_path = hf_hub_download(repo_id="Vongvathana/BI-LSTM", filename="emotion_model.h5")
        tokenizer_path = hf_hub_download(repo_id="Vongvathana/BI-LSTM", filename="tokenizer.pkl")
        ```
    *   **Keras model loading:**
        ```python
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        ```
    *   This deep-learning model analyzes raw texts up to **35 words** (`MAX_LENGTH = 35`) and returns a probability score for each of the **28 emotion classes**.

### 🏷️ LDA (NLP Topic Modeler)
*   **Source:** Stored locally offline within the `ml/lda_model` directory.
*   **Implementation (`ml/services/topic_modeler.py`):**
    *   Loads pre-compiled LDA dictionary and serialization matrices using `gensim`.
    *   Runs tokenization and stopwords cleansing without requiring any internet connection.

## 🤝 2. How the Models Work Together (The Core Pipeline)

The two models work in harmony inside the **`EntryAnalyzer`** class located in **`ml/services/entry_analyzer.py`**:

### 🔄 The 6-Step Processing Pipeline
When a user clicks **"Save Entry & Analyze"** in the Flutter app, the system executes this precise processing loop:

```
[ STEP 1 ] User Inputs Text (Flutter)
     │
     ▼
[ STEP 2 ] REST API Call (FastAPI Backend)
     │
     ├──────────────────────────────────────────────┐
     ▼                                              ▼
[ STEP 3a ] Bi-LSTM Pipeline                   [ STEP 3b ] LDA Pipeline
  - Tokenize & Pad Sequence                      - Text Cleaning (Stopwords removed)
  - Feed to Deep Learning Neural Net             - Convert to Bag-of-Words Vector
  - Output: Emotion (e.g. JOY)                   - Output: Theme (e.g. FAMILY)
     │                                              │
     └──────────────────────┬───────────────────────┘
                            ▼
                     [ STEP 4 ] Joint Explanation Generator
                       - Combined Insight generated
                            │
                            ▼
                     [ STEP 5 ] Save to Database (PostgreSQL)
                            │
                            ▼
                     [ STEP 6 ] Render UI (Flutter Dashboard)
```

1.  **Step 1: User Input (Flutter):** The user writes a journal entry in Flutter and clicks "Save & Analyze".
2.  **Step 2: API Route (FastAPI):** Flutter triggers an HTTP `POST` request to `/journal/entries` with the raw text.
3.  **Step 3a: Bi-LSTM Pipeline (Emotion):**
    *   **Preprocessing:** The text is tokenized and padded using `tokenizer.pkl` to a maximum sequence length of `35` words.
    *   **Inference:** The padded sequence is fed into **`mood_model.h5`**. The **Bi-LSTM** captures sequential contextual dependencies forward and backward.
    *   **Output:** The model outputs a GoEmotions class (e.g. `joy`) and computes an overall mood score between 0.0 and 1.0.
4.  **Step 3b: LDA Pipeline (Topic):**
    *   **Preprocessing:** Cleans text, removes common English stopwords (e.g. *the, is*), and lemmatizes words to base forms.
    *   **Inference:** Converts text to a Bag-of-Words (BoW) vector and feeds it to the **LDA** model to find keyword cluster probabilities.
    *   **Output:** Returns the dominant topic (e.g., `family`).
5.  **Step 4: Joint Insight Generation:** Merges both outputs using `_generate_combined_insight(emotion, topic)` to output a human-friendly correlated feedback sentence.
6.  **Step 5: DB Persistence:** Writes the entry text and all AI analysis attributes directly to the PostgreSQL database.
7.  **Step 6: Dashboard Display:** Returns the payload to Flutter, which renders the emotion badge (e.g., 😊), topic tag (e.g., 🏷️), and updates all analytics charts.

### 💻 Code Implementation:
```python
def analyze(self, text: str) -> Dict[str, Any]:
    # 1. Execute the Bi-LSTM Model to predict the emotion
    emotion, confidence = self.predictor.predict(text)
    mood_score = self.predictor.get_mood_score(emotion, confidence)
    
    # 2. Execute the Gensim LDA Model to extract the dominant theme
    topic, topic_conf = self.topic_modeler.get_dominant_topic(text)
    
    # 3. Integrate both results into a joint explanation!
    combined_insight = self._generate_combined_insight(emotion.value, str(topic))
    
    return {
        "emotion": emotion.value,
        "confidence": round(float(confidence), 2),
        "mood_score": round(float(mood_score), 2),
        "dominant_topic": str(topic),
        "topic_confidence": round(float(topic_conf), 2),
        "combined_insight": combined_insight
    }
```

### The Joint Explanation Generator
```python
def _generate_combined_insight(self, emotion: str, topic: str) -> str:
    # Maps emotional state to positive/negative categories and pairs it with the theme
    positive_emotions = ["joy", "love", "gratitude", "relief", ...]
    
    if emotion in positive_emotions:
        return f"The Bi-LSTM model detected a positive sentiment of '{emotion}' which aligns beautifully with your thoughts on '{topic}' (analyzed by LDA)."
    else:
        return f"The Bi-LSTM emotion predictor detected a '{emotion}' state, while the LDA topic modeler classified the theme as '{topic}'."
```

---

## 🚀 3. How the Pipeline is Implemented in the REST API

When a user writes a new journal entry in Flutter, it calls the `POST /api/v1/journal/entries` route. The request flow is controlled inside the backend routers:

### 1. The Controller (`backend/fastapi_server/app/routers/journal.py`)
This endpoint intercepts the user request, extracts the text, and routes it to the `insights_service`:

```python
@router.post("/entries", response_model=schemas.JournalEntry)
def create_entry(
    entry_in: schemas.JournalEntryCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    # Triggers the double-pipeline analysis
    analysis = insights_service.analyze_entry(entry_in.content)
    
    # Saves both the raw entry and its AI analysis into PostgreSQL
    db_entry = journal_repository.create(
        db, 
        obj_in=entry_in, 
        user_id=current_user.id,
        mood_score=analysis['mood_score'],
        sentiment_label=analysis['emotion'],
        dominant_topic=analysis['dominant_topic'],
        combined_insight=analysis['combined_insight']
    )
    return db_entry
```

### 2. The ML Config Router (`backend/fastapi_server/app/routers/ml.py`)
Controls and displays what models are currently active in our backend architecture:
```python
@router.get("/config")
def get_ml_config():
    return {
        "active_models": {
            "emotion_predictor": { "type": "keras", "class": "KerasPredictor" },
            "topic_modeler": { "type": "gensim", "class": "TopicModeler" }
        },
        "available_emotion_predictors": ["keras"], # Locked to Bi-LSTM
        "available_topic_modelers": ["gensim"]      # Locked to LDA
    }
```

---

## 📱 4. How the Frontend displays the Results (Flutter)

Our Flutter app contains two visual screens utilizing these API endpoints:

1.  **`lib/screens/journal_entry_screen.dart` (Submit Card):**
    *   Sends a `POST` request to `/journal/entries` with the raw text.
    *   Receives the computed AI payload.
    *   Opens an **AI Analysis Complete** modal displaying the **Bi-LSTM Emotion Badge**, the **LDA Topic Tag**, and the **Combined Joint Insight** text beautifully!
2.  **`lib/screens/insights_screen.dart` (Dashboard Analytics):**
    *   Calls `/ml/streaks` to fetch current streaks.
    *   Calls `/ml/time-of-day` and `/ml/weekly-patterns` to fetch daily/weekly mood cycles.
    *   Draws visual graphs (`fl_chart`) showing the user's emotional trends over time.

---

## 🔬 5. Model Specifications Deep Dive (For AI Defense)

When defending this project to Machine Learning evaluators or teachers, use these technical specifications:

### 🎭 Bidirectional LSTM (Emotion Engine)
*   **Architecture:** `Embedding(128-dim)` ──► `Bidirectional(LSTM(64))` ──► `Dropout(0.5)` ──► `Dense(28, activation='sigmoid')`.
*   **Why Sigmoid Activation?** GoEmotions is a **multi-label classification** task. A user can feel both `joy` and `surprise` at the same time. Softmax forces the output probabilities to sum to 1.0 (killing secondary emotions). Sigmoid treats each of the 28 emotions as an independent probability between 0 and 1.
*   **Tokenizer details:** Vocabulary built from 58,000 GoEmotions comments, using post-padding (`pad_sequences`) up to a maximum length (`MAX_LENGTH`) of **35 words**.

### 🏷️ Latent Dirichlet Allocation (Topic Engine)
*   **NLP Pipeline:** Tokenization ──► English Stopwords Removal ──► Lemmatization (base-form mapping, e.g. *studies* $\rightarrow$ *study*) ──► Bag-of-Words Vectorizer ──► LDA Classifier.
*   **Number of Topics:** Pre-trained on **20 topic clusters** mapping raw keyword frequencies to human-friendly tags (e.g. *Appearance & Opinion*, *Relationships & Affection*, *School & Education*).

---

## 🎤 6. The 5-Minute Presentation Script (Cheat-Sheet)

If you are selected to present this project, use this exact 5-minute script:

### ⏱️ Minute 1: The Problem & Solution (Intro)
> *"Good day, teacher and fellow classmates. Today we are presenting our **AI Mood Journal**. Traditional journals just record text, but our app acts as an active companion. We built a Flutter app connected to a FastAPI backend that runs **two separate AI models simultaneously**—a deep-learning **Bi-LSTM** to capture the user's emotion, and an NLP **LDA model** to extract what theme they are talking about. Let's see how they work together."*

### ⏱️ Minute 2: The UI Demonstration (Flutter Screen)
> *(Show your journal screen or the completed analysis modal)*
> *"When a user types their feelings and clicks 'Save & Analyze', Flutter fires a POST request to our API. The backend processes the input and returns our custom-designed **AI Analysis Complete** modal. 
> As you can see on this screen, the Bi-LSTM model successfully detected the emotion **'Love'** with 43% confidence, while the LDA Topic modeler correctly identified **'Love & Affection'** as the theme. We combine these two insights into a **Combined Joint Insight** paragraph so the user gets intelligent, human-like feedback!"*

### ⏱️ Minute 3: Under the Hood — Bi-LSTM (Deep Learning)
> *"Why did we choose **Bi-LSTM** for sentiment analysis? Language is read bidirectionally. To understand context, our model reads text both left-to-right and right-to-left. 
> It was trained on Google's **GoEmotions dataset** containing 58,000 comments. We used a **Sigmoid activation function** on the final dense layer because human feelings are mixed; it allows our model to detect multiple mixed emotions (like joy + relief) at the same time."*

### ⏱️ Minute 4: Under the Hood — LDA (Unsupervised NLP)
> *"In parallel, we run an **LDA (Latent Dirichlet Allocation) model** for topic extraction. LDA is an unsupervised mathematical classifier. It cleans the text, removes standard stopwords like 'the' and 'is', and assigns the entry to one of 20 pre-trained statistical word-topic distributions. 
> By keeping this offline and local inside Gensim, we ensure the app is fast, offline-friendly, and respects user privacy."*

### ⏱️ Minute 5: Modularity & Summary (Conclusion)
> *"To conclude, our application is built using the **Strategy Design Pattern**. The frontend and backend are decoupled, and the AI models run independently. This means we can swap or upgrade our AI engines without rebuilding the mobile app or server. 
> Thank you, and we are open to any questions!"*
