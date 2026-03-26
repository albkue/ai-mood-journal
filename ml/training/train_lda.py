"""
LDA Topic Model Training Script

This script trains an LDA model on journal entries to discover topics.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.topic_modeler import get_topic_modeler


def train_lda_from_entries(entries_texts: list, num_topics: int = 5):
    """
    Train LDA model on journal entries.
    
    Args:
        entries_texts: List of journal entry texts
        num_topics: Number of topics to discover
    
    Returns:
        True if training successful
    """
    print(f"Training LDA model with {num_topics} topics on {len(entries_texts)} entries...")
    
    modeler = get_topic_modeler(num_topics=num_topics, use_lda=False)
    success = modeler.train_lda(entries_texts)
    
    if success:
        print("✓ LDA model trained successfully!")
        
        # Print discovered topics
        topics_words = modeler.get_lda_topics_words(n_words=5)
        print("\nDiscovered Topics:")
        for topic, words in topics_words.items():
            print(f"  {topic}: {', '.join(words)}")
    else:
        print("✗ Failed to train LDA model")
    
    return success


def train_lda_from_database(num_topics: int = 5):
    """
    Train LDA model using entries from the database.
    Requires database connection.
    """
    try:
        # This would connect to your database and fetch entries
        # For now, placeholder
        print("Fetching entries from database...")
        
        # Example: Fetch all journal entries
        # from backend.fastapi_server.app.database import SessionLocal
        # from backend.fastapi_server.app.models import JournalEntry
        # db = SessionLocal()
        # entries = db.query(JournalEntry).all()
        # texts = [e.content for e in entries]
        
        print("Note: Connect to database to fetch real entries")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train LDA topic model')
    parser.add_argument('--topics', type=int, default=5, help='Number of topics (default: 5)')
    parser.add_argument('--from-db', action='store_true', help='Train from database entries')
    parser.add_argument('--sample', action='store_true', help='Train on sample data')
    
    args = parser.parse_args()
    
    if args.from_db:
        train_lda_from_database(args.topics)
    elif args.sample:
        # Sample journal entries for testing
        sample_entries = [
            "Today was a great day at work. I finished my project and my boss was impressed.",
            "Feeling stressed about the upcoming deadline. So much work to do.",
            "Had a wonderful dinner with my family. We laughed and shared stories.",
            "My workout was intense today. Feeling stronger and healthier.",
            "Worried about my finances this month. Need to budget better.",
            "Spent the weekend hiking with friends. Nature is so refreshing.",
            "Studying for my exams is exhausting. Can't wait for them to be over.",
            "My relationship is going through a rough patch. We need to communicate better.",
            "Started a new hobby - painting! It's so relaxing and creative.",
            "Doctor said my health is improving. Keep exercising and eating well.",
            "Work meeting was productive. We made good progress on the new feature.",
            "Missing my family back home. Need to call them more often.",
        ]
        train_lda_from_entries(sample_entries, args.topics)
    else:
        print("Usage:")
        print("  python train_lda.py --sample          # Train on sample data")
        print("  python train_lda.py --from-db         # Train from database")
        print("  python train_lda.py --topics 8        # Specify number of topics")


if __name__ == "__main__":
    main()
