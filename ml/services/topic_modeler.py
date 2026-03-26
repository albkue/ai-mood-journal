from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
import pickle
import os

class TopicModeler:
    """
    Topic modeling using LDA (Latent Dirichlet Allocation).
    Supports both Gensim and scikit-learn LDA models.
    """
    
    # Common topics in mood journals (for fallback/initialization)
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
    
    # Model type constants
    MODEL_GENSIM = "gensim"
    MODEL_SKLEARN = "sklearn"
    
    def __init__(self, num_topics: int = 20, use_lda: bool = False, model_type: str = "gensim"):
        self.num_topics = num_topics
        self.use_lda = use_lda
        self.model_type = model_type
        self.model = None
        self.vectorizer = None  # For sklearn
        self.dictionary = None  # For Gensim
        self.topic_names = []
        
        if use_lda:
            if model_type == self.MODEL_GENSIM:
                self._load_gensim_lda()
            else:
                self._load_or_train_sklearn_lda()
    
    def _load_gensim_lda(self):
        """Load Gensim LDA model"""
        try:
            from gensim import corpora, models
            
            # Path to your Gensim model
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lda_model")
            model_path = os.path.join(model_dir, "lda_20topics.model")
            dict_path = os.path.join(model_dir, "dictionary.dict")
            
            if os.path.exists(model_path) and os.path.exists(dict_path):
                print(f"Loading Gensim LDA model from {model_dir}...")
                
                # Load model and dictionary
                self.model = models.LdaModel.load(model_path)
                self.dictionary = corpora.Dictionary.load(dict_path)
                
                # Get number of topics from model
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
                
                print(f"✓ Gensim LDA loaded: {self.num_topics} topics")
            else:
                print(f"✗ Gensim LDA model not found at {model_dir}")
                print("  Using keyword-based fallback...")
                self.use_lda = False
                
        except ImportError:
            print("✗ Gensim not installed. Run: pip install gensim")
            self.use_lda = False
        except Exception as e:
            print(f"✗ Error loading Gensim LDA: {e}")
            self.use_lda = False
    
    def _load_or_train_sklearn_lda(self):
        """Load existing sklearn LDA model or train a new one"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        model_path = os.path.join(model_dir, "lda_model.pkl")
        vectorizer_path = os.path.join(model_dir, "lda_vectorizer.pkl")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            self.model = None
            self.vectorizer = None
    
    def _preprocess(self, text: str) -> str:
        """Simple text preprocessing for LDA"""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text for Gensim"""
        processed = self._preprocess(text)
        # Simple tokenization - split on whitespace
        tokens = processed.split()
        # Remove short words
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """
        Extract topics from a collection of journal entries.
        
        Returns:
            Dictionary of {topic_name: relevance_score}
        """
        if not texts:
            return {}
        
        # Use Gensim LDA
        if self.use_lda and self.model_type == self.MODEL_GENSIM and self.model is not None:
            return self._extract_topics_gensim(texts)
        
        # Use sklearn LDA
        if self.use_lda and self.model is not None and self.vectorizer is not None:
            return self._extract_topics_sklearn(texts)
        
        # Fallback to keyword-based extraction
        return self._extract_topics_keyword(texts)
    
    def _extract_topics_gensim(self, texts: List[str]) -> Dict[str, float]:
        """Extract topics using Gensim LDA model"""
        try:
            from gensim import corpora
            
            # Tokenize all texts
            tokenized = [self._tokenize(text) for text in texts]
            
            # Create bag of words corpus
            corpus = [self.dictionary.doc2bow(tokens) for tokens in tokenized]
            
            # Get topic distributions for all documents
            all_topic_scores = {}
            
            for doc_bow in corpus:
                # Get topic distribution for this document
                doc_topics = self.model.get_document_topics(doc_bow, minimum_probability=0.0)
                
                for topic_id, score in doc_topics:
                    topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
                    if topic_name not in all_topic_scores:
                        all_topic_scores[topic_name] = 0.0
                    all_topic_scores[topic_name] += score
            
            # Average across all documents
            num_docs = len(texts)
            all_topic_scores = {k: v/num_docs for k, v in all_topic_scores.items()}
            
            # Sort by score and return top topics
            sorted_topics = dict(sorted(all_topic_scores.items(), key=lambda x: x[1], reverse=True))
            return sorted_topics
            
        except Exception as e:
            print(f"Error in Gensim topic extraction: {e}")
            return self._extract_topics_keyword(texts)
    
    def _extract_topics_sklearn(self, texts: List[str]) -> Dict[str, float]:
        """Extract topics using sklearn LDA model"""
        processed_texts = [self._preprocess(text) for text in texts]
        doc_term_matrix = self.vectorizer.transform(processed_texts)
        topic_distribution = self.model.transform(doc_term_matrix)
        
        # Average topic distribution across all texts
        avg_distribution = topic_distribution.mean(axis=0)
        
        # Create topic score dictionary
        topic_scores = {}
        for idx, score in enumerate(avg_distribution):
            if idx < len(self.topic_names):
                topic_scores[self.topic_names[idx]] = float(score)
        
        return dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_topics_keyword(self, texts: List[str]) -> Dict[str, float]:
        """Fallback keyword-based topic extraction"""
        combined_text = ' '.join(texts).lower()
        
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # Normalize scores
        total = sum(topic_scores.values())
        if total > 0:
            topic_scores = {k: v/total for k, v in topic_scores.items()}
        
        return dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:self.num_topics])
    
    def get_dominant_topic(self, text: str) -> Tuple[str, float]:
        """Get the dominant topic for a single entry"""
        if self.use_lda and self.model_type == self.MODEL_GENSIM and self.model is not None:
            return self._get_dominant_topic_gensim(text)
        
        topics = self.extract_topics([text])
        if not topics:
            return "general", 1.0
        return list(topics.items())[0]
    
    def _get_dominant_topic_gensim(self, text: str) -> Tuple[str, float]:
        """Get dominant topic using Gensim model for single text"""
        try:
            tokens = self._tokenize(text)
            doc_bow = self.dictionary.doc2bow(tokens)
            doc_topics = self.model.get_document_topics(doc_bow)
            
            if doc_topics:
                # Get topic with highest probability
                dominant = max(doc_topics, key=lambda x: x[1])
                topic_id, score = dominant
                topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
                return topic_name, float(score)
        except Exception as e:
            pass
        
        return "general", 1.0
    
    def get_topic_keywords(self, topic_name: str) -> List[str]:
        """Get keywords for a specific topic"""
        # Try to get from Gensim model
        if self.model_type == self.MODEL_GENSIM and self.model is not None:
            try:
                # Extract topic ID from name (topic_0_word -> 0)
                parts = topic_name.split('_')
                if len(parts) >= 2:
                    topic_id = int(parts[1])
                    top_words = self.model.show_topic(topic_id, topn=5)
                    return [word for word, _ in top_words]
            except:
                pass
        
        return self.TOPIC_KEYWORDS.get(topic_name, [])
    
    def get_lda_topics_words(self, n_words: int = 5) -> Dict[str, List[str]]:
        """
        Get top words for each LDA topic.
        
        Returns:
            Dictionary of {topic_name: [top_words]}
        """
        if not self.use_lda:
            return {}
        
        if self.model_type == self.MODEL_GENSIM and self.model is not None:
            topics_words = {}
            for topic_id in range(self.num_topics):
                top_words = self.model.show_topic(topic_id, topn=n_words)
                topic_name = self.topic_names[topic_id] if topic_id < len(self.topic_names) else f"topic_{topic_id}"
                topics_words[topic_name] = [word for word, _ in top_words]
            return topics_words
        
        # sklearn model
        if self.model is not None and self.vectorizer is not None:
            feature_names = self.vectorizer.get_feature_names_out()
            topics_words = {}
            
            for topic_idx, topic in enumerate(self.model.components_):
                top_indices = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_name = self.topic_names[topic_idx] if topic_idx < len(self.topic_names) else f"topic_{topic_idx}"
                topics_words[topic_name] = top_words
            
            return topics_words
        
        return {}
    
    def train_sklearn_lda(self, texts: List[str]) -> bool:
        """
        Train sklearn LDA model on journal entries.
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            if len(texts) < 10:
                print("Not enough texts to train LDA (need at least 10)")
                return False
            
            processed_texts = [self._preprocess(text) for text in texts]
            
            self.vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=1000,
                stop_words='english'
            )
            doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
            
            self.model = LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=42,
                max_iter=10
            )
            self.model.fit(doc_term_matrix)
            
            feature_names = self.vectorizer.get_feature_names_out()
            self.topic_names = []
            for topic_idx, topic in enumerate(self.model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                self.topic_names.append(f"topic_{topic_idx}_{top_words[0]}")
            
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(model_dir, exist_ok=True)
            
            with open(os.path.join(model_dir, "lda_model.pkl"), 'wb') as f:
                pickle.dump(self.model, f)
            with open(os.path.join(model_dir, "lda_vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            self.use_lda = True
            self.model_type = self.MODEL_SKLEARN
            return True
            
        except Exception as e:
            print(f"Error training LDA: {e}")
            return False


# Singleton instance
_modeler = None

def get_topic_modeler(num_topics: int = 20, use_lda: bool = False, model_type: str = "gensim") -> TopicModeler:
    global _modeler
    if _modeler is None:
        _modeler = TopicModeler(num_topics, use_lda, model_type)
    return _modeler
