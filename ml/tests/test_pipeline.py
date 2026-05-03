"""
Test script for the ML pipeline with Strategy Pattern.

Tests:
1. Emotion Predictor - keyword fallback
2. Emotion Predictor - pre-trained BERT
3. Topic Modeler - Gensim LDA
4. Topic Modeler - LLM (Gemini)
5. Full Entry Analyzer pipeline
"""
import os
import sys
import time

# Add ml/ root to path so services/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Test journal entries
TEST_ENTRIES = [
    "I had a wonderful day today! Everything went perfectly at work and I'm so grateful for my team.",
    "Feeling really anxious about the exam tomorrow. I can't stop worrying about failing.",
    "Just got into a huge argument with my partner. I'm so angry and frustrated right now.",
    "Had a quiet evening reading a book. Nothing special, just a normal day.",
    "I'm so excited about the weekend trip! Can't wait to see the mountains!",
]


def test_keyword_predictor():
    """Test 1: Keyword Emotion Predictor (fallback)"""
    print("\n" + "=" * 60)
    print("TEST 1: Keyword Emotion Predictor")
    print("=" * 60)
    
    from services.emotion_predictor import KeywordEmotionPredictor
    predictor = KeywordEmotionPredictor()
    
    for entry in TEST_ENTRIES:
        emotion, confidence = predictor.predict(entry)
        distribution = predictor.predict_distribution(entry)
        mood = predictor.get_mood_score(emotion, confidence)
        print(f"\n  Text: {entry[:60]}...")
        print(f"  Emotion: {emotion.value} (confidence: {confidence})")
        print(f"  Mood Score: {mood:.2f}")
        if distribution:
            top3 = list(distribution.items())[:3]
            print(f"  Top 3: {top3}")
    
    print("\n  ✅ Keyword predictor works!")
    return True


def test_bert_predictor():
    """Test 2: Pre-trained BERT Emotion Predictor"""
    print("\n" + "=" * 60)
    print("TEST 2: Pre-trained BERT Emotion Predictor")
    print("=" * 60)
    
    from services.emotion_predictor import PretrainedBertPredictor
    # Use distilbert variant (maps to RoBERTa as lightweight option)
    predictor = PretrainedBertPredictor(model_variant="distilbert")
    
    if predictor.classifier is None:
        print("  ⚠️  BERT model not available (transformers/torch issue)")
        return False
    
    for entry in TEST_ENTRIES:
        emotion, confidence = predictor.predict(entry)
        distribution = predictor.predict_distribution(entry)
        mood = predictor.get_mood_score(emotion, confidence)
        print(f"\n  Text: {entry[:60]}...")
        print(f"  Emotion: {emotion.value} (confidence: {confidence})")
        print(f"  Mood Score: {mood:.2f}")
        if distribution:
            top3 = list(distribution.items())[:3]
            print(f"  Top 3: {top3}")
    
    print("\n  ✅ BERT predictor works!")
    return True


def test_gensim_topic_modeler():
    """Test 3: Gensim LDA Topic Modeler"""
    print("\n" + "=" * 60)
    print("TEST 3: Gensim LDA Topic Modeler")
    print("=" * 60)
    
    from services.topic_modeler import GensimLDATopicModeler
    modeler = GensimLDATopicModeler()
    
    for entry in TEST_ENTRIES:
        topic, confidence = modeler.get_dominant_topic(entry)
        topics = modeler.extract_topics([entry])
        top3 = dict(list(topics.items())[:3])
        print(f"\n  Text: {entry[:60]}...")
        print(f"  Dominant Topic: {topic} (confidence: {confidence:.3f})")
        print(f"  Top 3 Topics: {top3}")
    
    print("\n  ✅ Gensim LDA works!")
    return True


def test_llm_topic_modeler():
    """Test 4: LLM Topic Modeler (Gemini)"""
    print("\n" + "=" * 60)
    print("TEST 4: LLM Topic Modeler (Gemini)")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set, skipping LLM test")
        return False
    
    from services.topic_modeler import LLMTopicModeler
    modeler = LLMTopicModeler(provider="gemini", api_key=api_key)
    
    for entry in TEST_ENTRIES[:2]:  # Only test 2 to save API calls
        topic, confidence = modeler.get_dominant_topic(entry)
        topics = modeler.extract_topics([entry])
        print(f"\n  Text: {entry[:60]}...")
        print(f"  Dominant Topic: {topic} (confidence: {confidence})")
        print(f"  All Topics: {topics}")
    
    print("\n  ✅ LLM topic modeler works!")
    return True


def test_entry_analyzer():
    """Test 5: Full Entry Analyzer Pipeline"""
    print("\n" + "=" * 60)
    print("TEST 5: Full Entry Analyzer Pipeline")
    print("=" * 60)
    
    # Use keyword + gensim for quick test
    from services.emotion_predictor import reset_predictor
    from services.topic_modeler import reset_topic_modeler
    reset_predictor()
    reset_topic_modeler()
    
    os.environ["EMOTION_PREDICTOR_TYPE"] = "keyword"
    os.environ["TOPIC_MODELER_TYPE"] = "gensim"
    
    from services.entry_analyzer import get_entry_analyzer
    analyzer = get_entry_analyzer()
    
    for entry in TEST_ENTRIES:
        result = analyzer.analyze(entry)
        print(f"\n  Text: {entry[:60]}...")
        print(f"  Emotion: {result['emotion']} ({result['confidence']})")
        print(f"  Mood: {result['mood_score']} ({result['mood_category']})")
        print(f"  Topic: {result['dominant_topic']}")
        print(f"  Emotion Dist: {dict(list(result['emotion_distribution'].items())[:3])}")
        print(f"  Topics Dist: {dict(list(result['topics_distribution'].items())[:3])}")
    
    print("\n  ✅ Entry Analyzer pipeline works!")
    return True


def test_bert_full_pipeline():
    """Test 6: Full Pipeline with BERT + LLM"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline - BERT + LLM (Gemini)")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set, skipping")
        return False
    
    from services.emotion_predictor import reset_predictor
    from services.topic_modeler import reset_topic_modeler
    reset_predictor()
    reset_topic_modeler()
    
    os.environ["EMOTION_PREDICTOR_TYPE"] = "bert"
    os.environ["TOPIC_MODELER_TYPE"] = "llm"
    
    from services.entry_analyzer import get_entry_analyzer
    analyzer = get_entry_analyzer()
    
    # Test just 1 entry with the full pipeline
    entry = TEST_ENTRIES[0]
    print(f"\n  Text: {entry}")
    
    start = time.time()
    result = analyzer.analyze(entry)
    elapsed = time.time() - start
    
    print(f"  Emotion: {result['emotion']} ({result['confidence']})")
    print(f"  Mood: {result['mood_score']} ({result['mood_category']})")
    print(f"  Topic: {result['dominant_topic']}")
    print(f"  Emotion Distribution: {result['emotion_distribution']}")
    print(f"  Topics Distribution: {result['topics_distribution']}")
    print(f"  Time: {elapsed:.2f}s")
    
    print("\n  ✅ BERT + LLM pipeline works!")
    return True


if __name__ == "__main__":
    print("🧪 ML Pipeline Test Suite")
    print("=" * 60)
    
    # Load .env
    load_dotenv()
    
    results = {}
    
    # Run tests
    try:
        results["keyword"] = test_keyword_predictor()
    except Exception as e:
        print(f"\n  ❌ Keyword test failed: {e}")
        results["keyword"] = False
    
    try:
        results["bert"] = test_bert_predictor()
    except Exception as e:
        print(f"\n  ❌ BERT test failed: {e}")
        results["bert"] = False
    
    try:
        results["gensim"] = test_gensim_topic_modeler()
    except Exception as e:
        print(f"\n  ❌ Gensim test failed: {e}")
        results["gensim"] = False
    
    try:
        results["llm"] = test_llm_topic_modeler()
    except Exception as e:
        print(f"\n  ❌ LLM test failed: {e}")
        results["llm"] = False
    
    try:
        results["analyzer"] = test_entry_analyzer()
    except Exception as e:
        print(f"\n  ❌ Entry Analyzer test failed: {e}")
        results["analyzer"] = False
    
    try:
        results["full_pipeline"] = test_bert_full_pipeline()
    except Exception as e:
        print(f"\n  ❌ Full Pipeline test failed: {e}")
        results["full_pipeline"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed_count}/{len(results)} passed")
