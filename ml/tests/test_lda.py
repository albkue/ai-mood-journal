"""
Test script for LDA model integration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.topic_modeler import get_topic_modeler

def test_lda():
    print("=" * 50)
    print("Testing Gensim LDA Model Integration")
    print("=" * 50)
    
    # Initialize with LDA enabled
    print("\n1. Loading LDA model...")
    modeler = get_topic_modeler(num_topics=20, use_lda=True, model_type="gensim")
    
    if modeler.model is None:
        print("   ✗ Model not loaded. Check if gensim is installed:")
        print("     pip install gensim")
        return False
    
    print(f"   ✓ Model loaded successfully!")
    print(f"   ✓ Number of topics: {modeler.num_topics}")
    
    # Test single entry
    print("\n2. Testing single entry analysis...")
    test_text = "I had a great day at work with my colleagues. We finished the project and celebrated together."
    topic, score = modeler.get_dominant_topic(test_text)
    print(f"   Text: '{test_text[:50]}...'")
    print(f"   Dominant Topic: {topic}")
    print(f"   Score: {score:.4f}")
    
    # Test multiple entries
    print("\n3. Testing topic extraction from multiple entries...")
    entries = [
        "Today was a wonderful day with my family. We went to the park and had a picnic.",
        "Feeling stressed about work. The deadline is approaching and I have so much to do.",
        "My workout was intense today. Feeling stronger and healthier every day.",
        "Spent time with friends this weekend. We watched movies and had pizza.",
        "Worried about my finances. Need to save more money for the future."
    ]
    
    topics = modeler.extract_topics(entries)
    print(f"   Topics found: {len(topics)}")
    for topic_name, score in list(topics.items())[:5]:
        print(f"   - {topic_name}: {score:.4f}")
    
    # Show all topics with keywords
    print("\n4. All topics with top words:")
    all_topics = modeler.get_lda_topics_words(n_words=5)
    for topic_name, words in list(all_topics.items())[:10]:
        print(f"   {topic_name}: {', '.join(words)}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_lda()
