# System Diagrams

## High-Level Architecture

```mermaid
graph TB
    subgraph "Mobile App"
        A[Flutter Frontend]
        A1[Login Screen]
        A2[Home Screen]
        A3[Journal Entry]
        A4[Insights Screen]
    end

    subgraph "Backend Server"
        B[FastAPI]
        B1[Auth Router]
        B2[Journal Router]
        B3[ML Router]
    end

    subgraph "ML Pipeline"
        C[Text Preprocessor]
        D[Emotion Predictor<br/>GoEmotions]
        E[Topic Modeler<br/>LDA 20 topics]
        F[Insights Service]
    end

    subgraph "Data Layer"
        G[(PostgreSQL)]
    end

    A --> B
    A1 --> B1
    A2 --> B2
    A3 --> B2
    A4 --> B3

    B1 --> G
    B2 --> G
    B3 --> F

    F --> C
    C --> D
    C --> E
    D --> F
    E --> F
```

---

## ML Pipeline Flow

```mermaid
flowchart LR
    A[Journal Entry] --> B[Text Preprocessing]
    B --> C[Emotion Prediction]
    B --> D[Topic Modeling]
    C --> E[Mood Score]
    D --> F[Topic Distribution]
    E --> G[Insights Aggregation]
    F --> G
    G --> H[API Response]
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Flutter
    participant FastAPI
    participant ML Pipeline
    participant PostgreSQL

    User->>Flutter: Write journal entry
    Flutter->>FastAPI: POST /journal/
    FastAPI->>PostgreSQL: Save entry
    FastAPI->>ML Pipeline: Analyze text
    ML Pipeline->>ML Pipeline: Preprocess
    ML Pipeline->>ML Pipeline: Predict emotion
    ML Pipeline->>ML Pipeline: Extract topics
    ML Pipeline-->>FastAPI: Return analysis
    FastAPI->>PostgreSQL: Update entry with ML data
    FastAPI-->>Flutter: Return entry
    Flutter-->>User: Display entry with insights
```

---

## Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant Flutter
    participant FastAPI
    participant PostgreSQL

    User->>Flutter: Enter credentials
    Flutter->>FastAPI: POST /auth/login
    FastAPI->>PostgreSQL: Find user
    PostgreSQL-->>FastAPI: User data
    FastAPI->>FastAPI: Verify password (bcrypt)
    FastAPI->>FastAPI: Generate JWT token
    FastAPI-->>Flutter: Return token
    Flutter->>Flutter: Store token (SharedPreferences)
    Flutter-->>User: Navigate to home

    Note over Flutter,FastAPI: Subsequent requests include Authorization header
```

---

## Mood Insights Flow

```mermaid
sequenceDiagram
    participant User
    participant Flutter
    participant FastAPI
    participant ML Pipeline
    participant PostgreSQL

    User->>Flutter: Request insights
    Flutter->>FastAPI: GET /ml/insights
    FastAPI->>PostgreSQL: Fetch all user entries
    PostgreSQL-->>FastAPI: Return entries
    FastAPI->>ML Pipeline: Calculate insights
    ML Pipeline->>ML Pipeline: Aggregate daily emotions
    ML Pipeline->>ML Pipeline: Calculate mood trends
    ML Pipeline->>ML Pipeline: Compute statistics
    ML Pipeline-->>FastAPI: Return insights
    FastAPI-->>Flutter: Return MoodInsight object
    Flutter-->>User: Display charts and trends
```

---

## Database Schema

```mermaid
erDiagram
    USERS ||--o{ JOURNAL_ENTRIES : creates
    
    USERS {
        int id PK
        string email UK
        string username UK
        string hashed_password
        datetime created_at
    }
    
    JOURNAL_ENTRIES {
        int id PK
        int user_id FK
        string title
        text content
        int mood_rating
        float mood_score
        string sentiment_label
        string dominant_topic
        datetime created_at
        datetime updated_at
    }
```

---

## Repository Pattern

```mermaid
graph TB
    subgraph "API Layer"
        R[Router]
    end
    
    subgraph "Business Logic"
        S[Service]
    end
    
    subgraph "Data Access"
        RP[Repository]
    end
    
    subgraph "Database"
        DB[(PostgreSQL)]
    end
    
    R --> S
    S --> RP
    RP --> DB
```

---

## Emotion Categories

```mermaid
graph TB
    subgraph "High Positive (0.9)"
        E1[joy]
        E2[love]
        E3[gratitude]
        E4[excitement]
        E5[pride]
        E6[relief]
    end
    
    subgraph "Medium Positive (0.7)"
        E7[admiration]
        E8[amusement]
        E9[approval]
        E10[caring]
        E11[curiosity]
        E12[desire]
        E13[optimism]
        E14[realization]
    end
    
    subgraph "Neutral (0.5)"
        E15[neutral]
        E16[confusion]
        E17[surprise]
    end
    
    subgraph "Medium Negative (0.3)"
        E18[annoyance]
        E19[disappointment]
        E20[disapproval]
        E21[nervousness]
        E22[remorse]
    end
    
    subgraph "High Negative (0.1)"
        E23[anger]
        E24[disgust]
        E25[embarrassment]
        E26[fear]
        E27[grief]
        E28[sadness]
    end
```

---

## Docker Services

```mermaid
graph TB
    subgraph "Docker Compose"
        DB[(PostgreSQL<br/>:5432)]
        API[FastAPI Backend<br/>:8000]
    end
    
    subgraph "External"
        FL[Flutter App]
    end
    
    FL -->|HTTP| API
    API -->|SQL| DB
```
