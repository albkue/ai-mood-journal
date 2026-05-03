"""
Topic Modeler - Strategy Pattern implementation.

Supports multiple topic extraction backends:
- GensimLDATopicModeler: Traditional LDA (current, good for large corpora)
- LLMTopicModeler: Gemini/OpenAI-powered extraction (best labels, needs API)
- ZeroShotTopicModeler: Local zero-shot classification (privacy-friendly)

All implement the same TopicModelerBase interface, so the rest of the
system (EntryAnalyzer, InsightsAggregator) doesn't care which one is used.

Configuration via environment variables:
- TOPIC_MODELER_TYPE: "gensim" | "llm" | "zeroshot" (default: "gensim")
- LLM_PROVIDER: "gemini" | "openai" (default: "gemini")
- GEMINI_API_KEY / OPENAI_API_KEY: API keys for LLM provider
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
import pickle
import os


# ============================================================
# Abstract Base Class - The Interface
# ============================================================

class TopicModelerBase(ABC):
    """Abstract base class for all topic modelers."""
    
    # Common topic keywords (shared across implementations for fallback)
    TOPIC_KEYWORDS = {
        "work": ["work", "job", "career", "boss", "colleague", "office", "meeting", "deadline", "project"],
        "family": ["family", "parent", "mother", "father", "sister", "brother", "child", "kid", "home"],
        "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "husband", "wife", "love", "date"],
        "health": ["health", "sick", "doctor", "exercise", "gym", "workout", "sleep", "diet", "mental"],
        "social": ["friend", "party", "social", "hangout", "fun", "weekend", "event", "gathering"],
        "hobby": ["hobby", "game", "music", "movie", "book", "art", "craft", "sport", "travel"],
        "finance": ["money", "finance", "budget", "expense", "save", "debt", "salary", "income"],
        "education": ["school", "study", "class", "exam", "grade", "learn", "course", "homework"],
    }
    
    @abstractmethod
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """
        Extract topics from a collection of journal entries.
        
        Returns:
            Dictionary of {topic_name: relevance_score}
        """
        pass
    
    @abstractmethod
    def get_dominant_topic(self, text: str) -> Tuple[str, float]:
        """
        Get the dominant topic for a single entry.
        
        Returns:
            Tuple of (topic_name, confidence_score)
        """
        pass
    
    def _preprocess(self, text: str) -> str:
        """Common text preprocessing."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _extract_topics_keyword(self, texts: List[str], num_topics: int = 20) -> Dict[str, float]:
        """Fallback keyword-based topic extraction (shared by all implementations)."""
        combined_text = ' '.join(texts).lower()
        
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        total = sum(topic_scores.values())
        if total > 0:
            topic_scores = {k: v / total for k, v in topic_scores.items()}
        
        return dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:num_topics])


# ============================================================
# Gensim LDA Topic Modeler (Current Implementation)
# ============================================================

class GensimLDATopicModeler(TopicModelerBase):
    """
    Topic modeling using Gensim LDA.
    Good for large corpora, topics are stable over time.
    Topics are word-lists (e.g., "topic_7_time") rather than human-readable.
    """
    
    # Human-readable labels mapped from raw topic names
    TOPIC_LABELS = {
        "topic_0_good": "Positivity & Growth",
        "topic_1_thank": "Gratitude",
        "topic_2_like": "Preferences & Likes",
        "topic_3_look": "Appearance & Opinion",
        "topic_4_problem": "Problem Solving",
        "topic_5_youre": "Relationships",
        "topic_6_sorry": "Apology & Regret",
        "topic_7_time": "Time & Persistence",
        "topic_8_much": "Intentions & Play",
        "topic_9_name": "Identity & Judgment",
        "topic_10_well": "Work & Achievement",
        "topic_11_day": "Daily Life & Friends",
        "topic_12_dont": "Desires & Boundaries",
        "topic_13_one": "Self Improvement",
        "topic_14_people": "Social Thoughts",
        "topic_15_long": "Media & Curiosity",
        "topic_16_said": "Communication",
        "topic_17_thanks": "Appreciation",
        "topic_18_thats": "Agreement",
        "topic_19_love": "Love & Affection",
    }
    
    def __init__(self, num_topics: int = 20):
        self.num_topics = num_topics
        self.model = None
        self.dictionary = None
        self.topic_names = []
        self._load_model()
    
    def _load_model(self):
        """Load Gensim LDA model from disk."""
        try:
            from gensim import corpora, models
            
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lda_model")
            model_path = os.path.join(model_dir, "lda_20topics.model")
            dict_path = os.path.join(model_dir, "dictionary.dict")
            
            if os.path.exists(model_path) and os.path.exists(dict_path):
                print(f"Loading Gensim LDA model from {model_dir}...")
                
                self.model = models.LdaModel.load(model_path)
                self.dictionary = corpora.Dictionary.load(dict_path)
                self.num_topics = self.model.num_topics
                
                # Generate topic names from top words
                self.topic_names = []
                for topic_id in range(self.num_topics):
                    top_words = self.model.show_topic(topic_id, topn=1)
                    if top_words:
                        topic_label = f"topic_{topic_id}_{top_words[0][0]}"
                    else:
                        topic_label = f"topic_{topic_id}"
                    self.topic_names.append(topic_label)
                
                print(f"  Gensim LDA loaded: {self.num_topics} topics")
            else:
                print(f"  Gensim LDA model not found at {model_dir}, using keyword fallback")
                self.model = None
                
        except ImportError:
            print("  Gensim not installed. Run: pip install gensim")
            self.model = None
        except Exception as e:
            print(f"  Error loading Gensim LDA: {e}")
            self.model = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text for Gensim."""
        processed = self._preprocess(text)
        tokens = processed.split()
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """Extract topics using Gensim LDA model."""
        if not texts:
            return {}
        
        if self.model is None or self.dictionary is None:
            return self._extract_topics_keyword(texts, self.num_topics)
        
        try:
            from gensim import corpora
            
            tokenized = [self._tokenize(text) for text in texts]
            corpus = [self.dictionary.doc2bow(tokens) for tokens in tokenized]
            
            all_topic_scores = {}
            for doc_bow in corpus:
                doc_topics = self.model.get_document_topics(doc_bow, minimum_probability=0.0)
                for topic_id, score in doc_topics:
                    topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
                    readable_name = self._get_readable_name(topic_name)
                    if readable_name not in all_topic_scores:
                        all_topic_scores[readable_name] = 0.0
                    all_topic_scores[readable_name] += score
            
            num_docs = len(texts)
            all_topic_scores = {k: v / num_docs for k, v in all_topic_scores.items()}
            return dict(sorted(all_topic_scores.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error in Gensim topic extraction: {e}")
            return self._extract_topics_keyword(texts, self.num_topics)
    
    def _get_readable_name(self, raw_topic: str) -> str:
        """Convert raw topic name to human-readable label."""
        return self.TOPIC_LABELS.get(raw_topic, raw_topic.replace("topic_", "").replace("_", " ").title())
    
    def get_dominant_topic(self, text: str) -> Tuple[str, float]:
        """Get dominant topic using Gensim model for single text."""
        if self.model is None or self.dictionary is None:
            topics = self._extract_topics_keyword([text], self.num_topics)
            if not topics:
                return "general", 1.0
            return list(topics.items())[0]
        
        try:
            tokens = self._tokenize(text)
            doc_bow = self.dictionary.doc2bow(tokens)
            doc_topics = self.model.get_document_topics(doc_bow)
            
            if doc_topics:
                dominant = max(doc_topics, key=lambda x: x[1])
                topic_id, score = dominant
                topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
                return self._get_readable_name(topic_name), float(score)
        except Exception:
            pass
        
        return "general", 1.0
    
    def get_topic_keywords(self, topic_name: str) -> List[str]:
        """Get keywords for a specific topic."""
        if self.model is not None:
            try:
                parts = topic_name.split('_')
                if len(parts) >= 2:
                    topic_id = int(parts[1])
                    top_words = self.model.show_topic(topic_id, topn=5)
                    return [word for word, _ in top_words]
            except Exception:
                pass
        return self.TOPIC_KEYWORDS.get(topic_name, [])
    
    def get_lda_topics_words(self, n_words: int = 5) -> Dict[str, List[str]]:
        """Get top words for each LDA topic."""
        if self.model is None:
            return {}
        
        topics_words = {}
        for topic_id in range(self.num_topics):
            top_words = self.model.show_topic(topic_id, topn=n_words)
            topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
            topics_words[topic_name] = [word for word, _ in top_words]
        return topics_words


# ============================================================
# LLM-based Topic Modeler (Gemini / OpenAI)
# ============================================================

class LLMTopicModeler(TopicModelerBase):
    """
    Topic extraction using LLM APIs (Gemini or OpenAI).
    
    Pros:
    - Human-readable topic labels ("work stress", "relationship worries")
    - Works great on short text (even single entries)
    - More coherent, distinct topics
    
    Cons:
    - External API dependency (latency, cost, privacy)
    - Rate limits apply
    """
    
    # System prompt for topic extraction
    SYSTEM_PROMPT = """You are a mental health journal analyzer. Given journal entries, identify 1-5 relevant topics.

For each topic, provide:
- A short, human-readable label (2-3 words, e.g., "work stress", "family bonding")
- A confidence score (0.0 to 1.0)

Respond in this exact JSON format only:
{"topics": [{"label": "topic label", "confidence": 0.8}, ...]}

Important: Only return the JSON, no other text. Topics should be relevant to mental health and daily life."""

    def __init__(self, provider: str = "gemini", api_key: str = None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if self.provider == "gemini":
            return os.environ.get("GEMINI_API_KEY", "")
        elif self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        return ""
    
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """Extract topics using LLM."""
        if not texts:
            return {}
        
        if not self.api_key:
            print("  LLM API key not configured, using keyword fallback")
            return self._extract_topics_keyword(texts)
        
        try:
            combined_text = "\n---\n".join(texts[:5])  # Limit to 5 entries to control cost
            response = self._call_llm(combined_text)
            return self._parse_response(response)
        except Exception as e:
            print(f"  LLM topic extraction failed: {e}")
            return self._extract_topics_keyword(texts)
    
    def get_dominant_topic(self, text: str) -> Tuple[str, float]:
        """Get dominant topic using LLM."""
        topics = self.extract_topics([text])
        if not topics:
            return "general", 1.0
        return list(topics.items())[0]
    
    def _call_llm(self, text: str) -> str:
        """Call LLM API based on provider."""
        if self.provider == "gemini":
            return self._call_gemini(text)
        elif self.provider == "openai":
            return self._call_openai(text)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def _call_gemini(self, text: str) -> str:
        """Call Google Gemini API."""
        import requests
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{self.SYSTEM_PROMPT}\n\nJournal entries:\n{text}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 256
            }
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    
    def _call_openai(self, text: str) -> str:
        """Call OpenAI ChatGPT API."""
        import requests
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Journal entries:\n{text}"}
            ],
            "temperature": 0.3,
            "max_tokens": 256
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _parse_response(self, response: str) -> Dict[str, float]:
        """Parse LLM response into topic dict."""
        import json
        
        try:
            # Strip markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            topics = data.get("topics", [])
            
            result = {}
            for item in topics:
                label = item.get("label", "").strip().lower()
                confidence = float(item.get("confidence", 0.5))
                if label:
                    result[label] = confidence
            
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  Failed to parse LLM response: {e}")
            print(f"  Raw response: {response[:200]}")
            return {}


# ============================================================
# Zero-Shot Topic Modeler (Local, Privacy-Friendly)
# ============================================================

class ZeroShotTopicModeler(TopicModelerBase):
    """
    Topic extraction using zero-shot classification (facebook/bart-large-mnli).
    
    Pros:
    - Runs locally (no data leaves server)
    - Works on short text
    - Human-readable labels from candidate list
    - No API costs
    
    Cons:
    - Limited to predefined candidate topics
    - Requires ~1.5GB model download on first run
    - Slower than keyword matching (but faster than API calls)
    """
    
    # Candidate topics for mental health journaling
    DEFAULT_CANDIDATES = [
        "work stress", "family", "relationships", "health and fitness",
        "social life", "hobbies and creativity", "finances",
        "education and learning", "anxiety and worry", "gratitude and joy",
        "sleep quality", "self-improvement", "loneliness", "accomplishments",
        "nature and outdoors", "food and cooking", "travel", "spirituality"
    ]
    
    def __init__(self, candidate_topics: List[str] = None, model_name: str = "facebook/bart-large-mnli"):
        self.candidate_topics = candidate_topics or self.DEFAULT_CANDIDATES
        self.model_name = model_name
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load zero-shot classification model."""
        try:
            from transformers import pipeline
            print(f"Loading zero-shot classifier ({self.model_name})...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # CPU; use 0 for GPU
            )
            print(f"  Zero-shot classifier loaded with {len(self.candidate_topics)} candidate topics")
        except ImportError:
            print("  transformers not installed. Run: pip install transformers torch")
            self.classifier = None
        except Exception as e:
            print(f"  Error loading zero-shot model: {e}")
            self.classifier = None
    
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """Extract topics using zero-shot classification."""
        if not texts:
            return {}
        
        if self.classifier is None:
            return self._extract_topics_keyword(texts)
        
        try:
            combined_text = " ".join(texts)
            # Truncate very long text to avoid OOM
            if len(combined_text) > 2000:
                combined_text = combined_text[:2000]
            
            result = self.classifier(
                combined_text,
                self.candidate_topics,
                multi_label=True
            )
            
            topic_scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                if score > 0.05:  # Filter very low scores
                    topic_scores[label] = round(score, 3)
            
            return dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"  Zero-shot classification failed: {e}")
            return self._extract_topics_keyword(texts)
    
    def get_dominant_topic(self, text: str) -> Tuple[str, float]:
        """Get dominant topic using zero-shot classification."""
        if self.classifier is None:
            topics = self._extract_topics_keyword([text])
            if not topics:
                return "general", 1.0
            return list(topics.items())[0]
        
        try:
            result = self.classifier(
                text[:2000],  # Truncate
                self.candidate_topics,
                multi_label=False
            )
            
            if result["labels"]:
                return result["labels"][0], round(result["scores"][0], 3)
        except Exception:
            pass
        
        return "general", 1.0


# ============================================================
# Factory Function - Environment-Configured Selection
# ============================================================

_modeler = None

def get_topic_modeler(
    model_type: str = None,
    num_topics: int = 20,
    **kwargs
) -> TopicModelerBase:
    """
    Factory function to get the appropriate topic modeler.
    
    Selects based on:
    1. Explicit model_type parameter
    2. TOPIC_MODELER_TYPE environment variable
    3. Default: "gensim"
    
    Args:
        model_type: "gensim" | "llm" | "zeroshot"
        num_topics: Number of topics (only used by Gensim LDA)
        **kwargs: Additional args passed to modeler constructor
    
    Returns:
        TopicModelerBase implementation
    """
    global _modeler
    if _modeler is not None:
        return _modeler
    
    # Determine model type
    if model_type is None:
        model_type = os.environ.get("TOPIC_MODELER_TYPE", "gensim").lower()
    
    print(f"Initializing TopicModeler (type={model_type})...")
    
    if model_type == "gensim":
        _modeler = GensimLDATopicModeler(num_topics=num_topics)
    
    elif model_type == "llm":
        provider = kwargs.get("provider") or os.environ.get("LLM_PROVIDER", "gemini").lower()
        api_key = kwargs.get("api_key")
        _modeler = LLMTopicModeler(provider=provider, api_key=api_key)
    
    elif model_type == "zeroshot":
        candidates = kwargs.get("candidate_topics")
        model_name = kwargs.get("model_name", "facebook/bart-large-mnli")
        _modeler = ZeroShotTopicModeler(candidate_topics=candidates, model_name=model_name)
    
    else:
        print(f"  Unknown model type '{model_type}', falling back to gensim")
        _modeler = GensimLDATopicModeler(num_topics=num_topics)
    
    return _modeler


def reset_topic_modeler():
    """Reset singleton (useful for testing or switching modelers)."""
    global _modeler
    _modeler = None
