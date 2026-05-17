import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/journal_entry.dart';
import '../services/api_service.dart';

class EntriesListScreen extends StatefulWidget {
  const EntriesListScreen({super.key});

  @override
  State<EntriesListScreen> createState() => _EntriesListScreenState();
}

class _EntriesListScreenState extends State<EntriesListScreen> {
  List<JournalEntry> _entries = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadEntries();
  }

  Future<void> _loadEntries() async {
    setState(() => _isLoading = true);
    try {
      final response = await ApiService.get('/journal/entries');
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        setState(() {
          _entries = data.map((json) => JournalEntry.fromJson(json)).toList();
          _isLoading = false;
        });
      } else {
        setState(() => _isLoading = false);
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load entries: $e')),
        );
      }
    }
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  String _getEmotionEmoji(String? label) {
    if (label == null) return '😐';
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

  Color _getMoodColor(double? score) {
    if (score == null) return Colors.grey;
    if (score >= 0.8) return Colors.green;
    if (score >= 0.5) return Colors.lightGreen;
    if (score >= 0.3) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_entries.isEmpty) {
      return RefreshIndicator(
        onRefresh: _loadEntries,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          child: Container(
            height: MediaQuery.of(context).size.height * 0.7,
            alignment: Alignment.center,
            child: const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.book, size: 64, color: Colors.grey),
                SizedBox(height: 16),
                Text(
                  'No entries yet',
                  style: TextStyle(fontSize: 18, color: Colors.grey, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  'Start writing your first journal entry!',
                  style: TextStyle(color: Colors.grey),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadEntries,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _entries.length,
        itemBuilder: (context, index) {
          final entry = _entries[index];
          final emoji = _getEmotionEmoji(entry.sentimentLabel);
          final moodColor = _getMoodColor(entry.moodScore);

          return Card(
            margin: const EdgeInsets.only(bottom: 16),
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [

                      Text(
                        _formatDate(entry.createdAt),
                        style: const TextStyle(color: Colors.grey, fontWeight: FontWeight.bold),
                      ),
                      if (entry.analysisStatus == 'completed') ...[
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: moodColor.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            children: [
                              Text(
                                emoji,
                                style: const TextStyle(fontSize: 16),
                              ),
                              const SizedBox(width: 4),
                              Text(
                                entry.sentimentLabel?.toUpperCase() ?? 'NEUTRAL',
                                style: TextStyle(
                                  color: moodColor,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ] else ...[
                        const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    entry.content,
                    style: const TextStyle(fontSize: 16, height: 1.4),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      if (entry.dominantTopic != null && entry.analysisStatus == 'completed')
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.deepPurple[50],
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.tag, size: 14, color: Colors.deepPurple),
                              const SizedBox(width: 4),
                              Text(
                                entry.dominantTopic!,
                                style: const TextStyle(
                                  color: Colors.deepPurple,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        )
                      else
                        const Text(
                          'Analyzing theme...',
                          style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic, fontSize: 12),
                        ),
                      if (entry.moodScore != null)
                        Text(
                          'Mood: ${(entry.moodScore! * 100).toStringAsFixed(0)}%',
                          style: TextStyle(
                            color: moodColor,
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
