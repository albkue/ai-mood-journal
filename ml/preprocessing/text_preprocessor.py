import re
import string
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        text = self._lowercase(text)
        text = self._remove_urls(text)
        text = self._remove_mentions(text)
        text = self._remove_hashtags(text)
        text = self._remove_punctuation(text)
        text = self._remove_numbers(text)
        text = self._remove_extra_whitespace(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return ' '.join(tokens)
    
    def _lowercase(self, text: str) -> str:
        return text.lower()
    
    def _remove_urls(self, text: str) -> str:
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    def _remove_mentions(self, text: str) -> str:
        return re.sub(r'@\w+', '', text)
    
    def _remove_hashtags(self, text: str) -> str:
        return re.sub(r'#\w+', '', text)
    
    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_numbers(self, text: str) -> str:
        return re.sub(r'\d+', '', text)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """Convenience function for quick preprocessing"""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)
