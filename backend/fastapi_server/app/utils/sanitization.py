"""
Input sanitization utilities for the Mood Journal API.

- HTML tag stripping
- String length validation
- Whitespace normalization
"""
import re
from typing import Optional


# Regex to match HTML tags
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
# Regex to match script-like patterns
SCRIPT_REGEX = re.compile(r'(javascript:|on\w+\s*=|<script|</script)', re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return HTML_TAG_REGEX.sub('', text)


def sanitize_string(text: str, max_length: int = 10000) -> str:
    """
    Sanitize a string input:
    1. Strip HTML tags
    2. Normalize whitespace
    3. Truncate to max_length
    4. Remove script-like patterns
    """
    if not text:
        return text
    
    # Strip HTML tags
    cleaned = strip_html(text)
    
    # Remove script-like patterns
    cleaned = SCRIPT_REGEX.sub('', cleaned)
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Truncate
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned


def sanitize_content(content: str, max_length: int = 50000) -> str:
    """
    Sanitize journal entry content.
    Less aggressive than sanitize_string - preserves newlines for readability.
    """
    if not content:
        return content
    
    # Strip HTML tags
    cleaned = strip_html(content)
    
    # Remove script-like patterns
    cleaned = SCRIPT_REGEX.sub('', cleaned)
    
    # Normalize excessive whitespace but preserve single newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
    cleaned = re.sub(r' {2,}', ' ', cleaned)  # Collapse multiple spaces
    
    # Truncate
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned.strip()


def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
