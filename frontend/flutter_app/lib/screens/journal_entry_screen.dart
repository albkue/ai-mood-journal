import 'dart:convert';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class JournalEntryScreen extends StatefulWidget {
  const JournalEntryScreen({super.key});

  @override
  State<JournalEntryScreen> createState() => _JournalEntryScreenState();
}

class _JournalEntryScreenState extends State<JournalEntryScreen> {
  final _contentController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _contentController.dispose();
    super.dispose();
  }

  String _getEmotionEmoji(String label) {
    final emotionMap = {
      'joy': '😊',
      'love': '❤️',
      'gratitude': '🙏',
      'excitement': '🤩',
      'pride': '🦁',
      'relief': '😌',
      'admiration': '👏',
      'amusement': '😄',
      'approval': '👍',
      'caring': '🤗',
      'curiosity': '🤔',
      'desire': '💫',
      'optimism': '🌅',
      'realization': '💡',
      'confusion': '😕',
      'surprise': '😮',
      'annoyance': '😒',
      'disappointment': '😞',
      'disapproval': '👎',
      'nervousness': '😰',
      'remorse': '🥺',
      'anger': '😡',
      'disgust': '🤢',
      'embarrassment': '😳',
      'fear': '😨',
      'grief': '😢',
      'sadness': '😭',
      'neutral': '😐',
    };
    return emotionMap[label.toLowerCase()] ?? '😐';
  }

  void _showAnalysisResultModal({
    required BuildContext context,
    required String emotion,
    required double confidence,
    required double moodScore,
    required String dominantTopic,
    required double topicConfidence,
    required String combinedInsight,
  }) {
    final emoji = _getEmotionEmoji(emotion);
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: const Row(
            children: [
              Icon(Icons.psychology, color: Colors.deepPurple, size: 28),
              SizedBox(width: 8),
              Text(
                'AI Analysis Complete',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
          ),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  'Here is how your models analyzed your entry:',
                  style: TextStyle(color: Colors.grey, fontSize: 13),
                ),
                const SizedBox(height: 16),
                
                // Model 1: Bi-LSTM
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.green.withOpacity(0.2)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.bubble_chart, color: Colors.green, size: 18),
                          SizedBox(width: 6),
                          Text(
                            'Bi-LSTM Emotion Model',
                            style: TextStyle(fontWeight: FontWeight.bold, color: Colors.green, fontSize: 12),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Text(emoji, style: const TextStyle(fontSize: 28)),
                          const SizedBox(width: 10),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                emotion.toUpperCase(),
                                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                              ),
                              Text(
                                'Confidence: ${(confidence * 100).toStringAsFixed(1)}% | Mood: ${(moodScore * 100).toStringAsFixed(0)}%',
                                style: const TextStyle(color: Colors.grey, fontSize: 11),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                
                // Model 2: Gensim LDA
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.deepPurple.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.deepPurple.withOpacity(0.2)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.topic, color: Colors.deepPurple, size: 18),
                          SizedBox(width: 6),
                          Text(
                            'Gensim LDA Topic Modeler',
                            style: TextStyle(fontWeight: FontWeight.bold, color: Colors.deepPurple, fontSize: 12),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          const Icon(Icons.tag, size: 28, color: Colors.deepPurple),
                          const SizedBox(width: 10),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                dominantTopic,
                                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                              ),
                              Text(
                                'Topic relevance: ${(topicConfidence * 100).toStringAsFixed(1)}%',
                                style: const TextStyle(color: Colors.grey, fontSize: 11),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
                
                // Combined Insight Card
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.amber.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.amber.withOpacity(0.3)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.auto_awesome, color: Colors.amber, size: 18),
                          SizedBox(width: 6),
                          Text(
                            'Combined Joint Insight',
                            style: TextStyle(fontWeight: FontWeight.bold, color: Colors.amber, fontSize: 12),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        combinedInsight,
                        style: const TextStyle(fontSize: 13, height: 1.4),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Great!', style: TextStyle(fontWeight: FontWeight.bold)),
            ),
          ],
        );
      },
    );
  }

  Future<void> _saveEntry() async {
    final text = _contentController.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please write something')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      // 1. Persist the entry to the Database via REST API
      final saveResponse = await ApiService.post('/journal/entries', {
        'content': text,
      });

      if (saveResponse.statusCode == 200) {
        // 2. Perform instant live analysis on the models
        final analyzeResponse = await ApiService.post('/ml/analyze', {
          'text': text,
        });

        setState(() => _isLoading = false);

        if (analyzeResponse.statusCode == 200 && mounted) {
          final res = jsonDecode(analyzeResponse.body);
          
          _contentController.clear();
          
          _showAnalysisResultModal(
            context: context,
            emotion: res['emotion'] ?? 'neutral',
            confidence: (res['confidence'] ?? 1.0).toDouble(),
            moodScore: (res['mood_score'] ?? 0.5).toDouble(),
            dominantTopic: res['dominant_topic'] ?? 'General',
            topicConfidence: (res['topic_confidence'] ?? 1.0).toDouble(),
            combinedInsight: res['combined_insight'] ?? 'Analysis complete.',
          );
        } else if (mounted) {
          _contentController.clear();
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Entry saved successfully!')),
          );
        }
      } else {
        setState(() => _isLoading = false);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to save entry: ${saveResponse.statusCode}')),
          );
        }
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving entry: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'How are you feeling today?',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: TextField(
              controller: _contentController,
              maxLines: null,
              expands: true,
              textAlignVertical: TextAlignVertical.top,
              decoration: InputDecoration(
                hintText: 'Write about your day, your thoughts, your feelings...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                filled: true,
                fillColor: Colors.grey[100],
              ),
            ),
          ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: _isLoading ? null : _saveEntry,
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              backgroundColor: Colors.deepPurple,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
            child: _isLoading
                ? const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                  )
                : const Text('Save Entry & Analyze', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }
}
