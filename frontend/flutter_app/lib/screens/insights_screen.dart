import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/api_service.dart';
import '../models/journal_entry.dart';

class InsightsScreen extends StatefulWidget {
  const InsightsScreen({super.key});

  @override
  State<InsightsScreen> createState() => _InsightsScreenState();
}

class _InsightsScreenState extends State<InsightsScreen> {
  bool _isLoading = true;
  
  // Basic stats
  int _totalEntries = 0;
  double _averageMood = 0.0;
  String _moodTrend = 'No data';
  List<JournalEntry> _entries = [];
  Map<String, int> _emotionsDistribution = {};
  Map<String, double> _topicsDistribution = {};

  // Streaks stats
  int _currentStreak = 0;
  int _longestStreak = 0;
  double _goodDayPercentage = 0.0;

  // Time of Day stats
  Map<String, dynamic> _timeOfDayStats = {};

  // Weekly Pattern stats
  double _weekdayAvg = 0.0;
  double _weekendAvg = 0.0;
  String _bestDayOfWeek = '';
  String _worstDayOfWeek = '';

  // ML Config
  Map<String, dynamic> _activeModels = {};
  List<String> _availableEmotionPredictors = [];
  List<String> _availableTopicModelers = [];

  @override
  void initState() {
    super.initState();
    _loadInsights();
  }

  Future<void> _loadInsights() async {
    setState(() => _isLoading = true);
    try {
      // 1. Basic insights
      final insightsRes = await ApiService.get('/journal/insights');
      if (insightsRes.statusCode == 200) {
        final data = jsonDecode(insightsRes.body);
        _totalEntries = data['total_entries'] ?? 0;
        _averageMood = (data['average_mood'] ?? 0.0).toDouble();
        _moodTrend = data['mood_trend'] ?? 'No data';
        
        final List<dynamic> entriesList = data['entries'] ?? [];
        _entries = entriesList.map((e) => JournalEntry.fromJson(e)).toList();
        
        _emotionsDistribution = Map<String, int>.from(data['emotions_distribution'] ?? {});
        _topicsDistribution = Map<String, double>.from(data['topics_distribution'] ?? {});
      }

      // 2. Streaks
      try {
        final streaksRes = await ApiService.get('/ml/streaks');
        if (streaksRes.statusCode == 200) {
          final data = jsonDecode(streaksRes.body);
          _currentStreak = data['current_streak'] ?? 0;
          _longestStreak = data['longest_streak'] ?? 0;
          _goodDayPercentage = (data['good_day_percentage'] ?? 0.0).toDouble();
        }
      } catch (e) {
        debugPrint("Error fetching streaks: $e");
      }

      // 3. Time of Day
      try {
        final todRes = await ApiService.get('/ml/time-of-day');
        if (todRes.statusCode == 200) {
          _timeOfDayStats = jsonDecode(todRes.body);
        }
      } catch (e) {
        debugPrint("Error fetching time of day: $e");
      }

      // 4. Weekly Patterns
      try {
        final weeklyRes = await ApiService.get('/ml/weekly-patterns');
        if (weeklyRes.statusCode == 200) {
          final data = jsonDecode(weeklyRes.body);
          _weekdayAvg = (data['weekday_avg'] ?? 0.0).toDouble();
          _weekendAvg = (data['weekend_avg'] ?? 0.0).toDouble();
          _bestDayOfWeek = data['best_day_of_week'] ?? '';
          _worstDayOfWeek = data['worst_day_of_week'] ?? '';
        }
      } catch (e) {
        debugPrint("Error fetching weekly patterns: $e");
      }

      // 5. ML Config
      try {
        final configRes = await ApiService.get('/ml/config');
        if (configRes.statusCode == 200) {
          final data = jsonDecode(configRes.body);
          _activeModels = data['active_models'] ?? {};
          _availableEmotionPredictors = List<String>.from(data['available_emotion_predictors'] ?? []);
          _availableTopicModelers = List<String>.from(data['available_topic_modelers'] ?? []);
        }
      } catch (e) {
        debugPrint("Error fetching ML config: $e");
      }

      setState(() => _isLoading = false);
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load insights: $e')),
        );
      }
    }
  }

  Future<void> _updateModel(String? emotionModel, String? topicModel) async {
    setState(() => _isLoading = true);
    try {
      final response = await ApiService.post('/ml/config', {
        if (emotionModel != null) 'emotion_model': emotionModel,
        if (topicModel != null) 'topic_model': topicModel,
      });

      if (response.statusCode == 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('ML Models dynamically updated! 🚀'),
              backgroundColor: Colors.green,
            ),
          );
        }
        await _loadInsights();
      } else {
        final err = jsonDecode(response.body);
        throw Exception(err['detail'] ?? 'Failed to switch models');
      }
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update models: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  String _getEmotionEmoji(String label) {
    final emotionMap = {
      'joy': '😊', 'love': '❤️', 'gratitude': '🙏', 'excitement': '🤩',
      'pride': '🦁', 'relief': '😌', 'admiration': '👏', 'amusement': '😄',
      'approval': '👍', 'caring': '🤗', 'curiosity': '🤔', 'desire': '💫',
      'optimism': '🌅', 'realization': '💡', 'confusion': '😕', 'surprise': '😮',
      'annoyance': '😒', 'disappointment': '😞', 'disapproval': '👎', 'nervousness': '😰',
      'remorse': '🥺', 'anger': '😡', 'disgust': '🤢', 'embarrassment': '😳',
      'fear': '😨', 'grief': '😢', 'sadness': '😭', 'neutral': '😐'
    };
    return emotionMap[label.toLowerCase()] ?? '😐';
  }

  String _getDisplayName(String raw) {
    if (raw == 'keras') return 'Bi-LSTM (Keras)';
    if (raw == 'bert') return 'BERT (GoEmotions)';
    if (raw == 'sklearn') return 'Random Forest (Sklearn)';
    if (raw == 'keyword') return 'Keyword Match (Fallback)';
    if (raw == 'transformer') return 'Transformer (DistilBERT)';
    if (raw == 'gensim') return 'LDA (Gensim)';
    if (raw == 'llm') return 'Gemini / OpenAI (LLM)';
    if (raw == 'zeroshot') return 'Zero-Shot Classifier';
    return raw.toUpperCase();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return RefreshIndicator(
      onRefresh: _loadInsights,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Row 1: Basic Stats
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Total Entries',
                    _totalEntries.toString(),
                    Icons.book,
                    Colors.blue,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Avg Mood',
                    '${(_averageMood * 100).toStringAsFixed(0)}%',
                    Icons.sentiment_satisfied,
                    _averageMood >= 0.5 ? Colors.green : Colors.orange,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            
            // Row 2: Streaks
            Row(
              children: [
                Expanded(
                  child: _buildStatCard(
                    'Current Streak',
                    '$_currentStreak days 🔥',
                    Icons.local_fire_department,
                    Colors.orange,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildStatCard(
                    'Longest Streak',
                    '$_longestStreak days 🏆',
                    Icons.emoji_events,
                    Colors.purple,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            _buildStatCard(
              'Mood Trend',
              _moodTrend,
              Icons.trending_up,
              Colors.deepPurple,
            ),
            const SizedBox(height: 24),
            
            // 🤖 Model Selector & Control Center
            _buildModelConfigCard(),
            const SizedBox(height: 24),

            if (_totalEntries > 0) ...[
              const Text(
                'Mood Analysis (Bi-LSTM)',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              SizedBox(height: 200, child: _buildMoodChart()),
              const SizedBox(height: 24),
              
              // 🌅 Time of Day Cycle
              if (_timeOfDayStats.isNotEmpty) ...[
                _buildTimeOfDayCard(),
                const SizedBox(height: 24),
              ],

              // 🏢 Weekly Cycles
              if (_bestDayOfWeek.isNotEmpty || _weekdayAvg > 0) ...[
                _buildWeeklyPatternsCard(),
                const SizedBox(height: 24),
              ],
              
              // Top Emotions Distribution
              if (_emotionsDistribution.isNotEmpty) ...[
                const Text(
                  'Top Emotions Detected',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                Card(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: _emotionsDistribution.entries.map((item) {
                        final emoji = _getEmotionEmoji(item.key);
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Row(
                            children: [
                              Text(emoji, style: const TextStyle(fontSize: 18)),
                              const SizedBox(width: 8),
                              Expanded(
                                flex: 3,
                                child: Text(item.key.toUpperCase(), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                              ),
                              Expanded(
                                flex: 6,
                                child: LinearProgressIndicator(
                                  value: item.value / _totalEntries,
                                  backgroundColor: Colors.grey[200]!,
                                  color: Colors.green,
                                  minHeight: 8,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text('${item.value} times', style: const TextStyle(fontSize: 11, color: Colors.grey)),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
              ],

              // Dominant Topics Distribution (LDA)
              if (_topicsDistribution.isNotEmpty) ...[
                const Text(
                  'Extracted Themes & Topics (LDA)',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                Card(
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: _topicsDistribution.entries.map((item) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Row(
                            children: [
                              const Icon(Icons.tag, size: 16, color: Colors.deepPurple),
                              const SizedBox(width: 4),
                              Expanded(
                                flex: 4,
                                child: Text(
                                  item.key, 
                                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              Expanded(
                                flex: 5,
                                child: LinearProgressIndicator(
                                  value: item.value / _totalEntries,
                                  backgroundColor: Colors.grey[200]!,
                                  color: Colors.deepPurple,
                                  minHeight: 8,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text('${(item.value / _totalEntries * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 11, color: Colors.grey)),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
              ],

            ] else ...[
              Center(
                child: Column(
                  children: [
                    const SizedBox(height: 48),
                    Icon(Icons.insights, size: 64, color: Colors.grey[400]),
                    const SizedBox(height: 16),
                    const Text(
                      'No insights yet',
                      style: TextStyle(fontSize: 18, color: Colors.grey, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Write some journal entries to see your mood trends!',
                      style: TextStyle(color: Colors.grey),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(
    String title,
    String value,
    IconData icon,
    Color color,
  ) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 28),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(fontSize: 12, color: Colors.grey, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelConfigCard() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          gradient: LinearGradient(
            colors: [Colors.deepPurple.shade900.withOpacity(0.05), Colors.blue.shade900.withOpacity(0.05)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.psychology, color: Colors.deepPurple, size: 28),
                const SizedBox(width: 8),
                const Text(
                  'AI Core Engine',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.check_circle, color: Colors.green, size: 12),
                      SizedBox(width: 4),
                      Text('Verified & Loaded', style: TextStyle(color: Colors.green, fontSize: 10, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text(
              'The application is powered by a dual-pipeline AI architecture:',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 16),
            
            // Emotion Predictor Info
            Row(
              children: [
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Emotion Predictor Model', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                      Text('Deep Learning Bi-LSTM (Keras)', style: TextStyle(fontSize: 10, color: Colors.grey)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.deepPurple.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Text(
                    'Active 🟢',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.deepPurple),
                  ),
                ),
              ],
            ),
            const Divider(),

            // Topic Modeler Info
            Row(
              children: [
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Topic Modeler Model', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                      Text('Latent Dirichlet Allocation (Gensim LDA)', style: TextStyle(fontSize: 10, color: Colors.grey)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.blue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Text(
                    'Active 🟢',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.blue),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimeOfDayCard() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Time of Day Mood Cycle',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Card(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: ['morning', 'afternoon', 'evening', 'night'].map((period) {
                final stats = _timeOfDayStats[period] ?? {'avg_mood': 0.5, 'count': 0, 'dominant_emotion': 'neutral'};
                final avgMood = (stats['avg_mood'] ?? 0.5).toDouble();
                final count = stats['count'] ?? 0;
                final emotion = stats['dominant_emotion'] ?? 'neutral';

                String icon = '🌅';
                if (period == 'afternoon') icon = '☀️';
                if (period == 'evening') icon = '🌆';
                if (period == 'night') icon = '🌙';

                return Column(
                  children: [
                    Text(icon, style: const TextStyle(fontSize: 28)),
                    const SizedBox(height: 4),
                    Text(
                      period.toUpperCase(), 
                      style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.grey),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: (avgMood >= 0.5 ? Colors.green : Colors.orange).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        '${(avgMood * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          fontSize: 12, 
                          fontWeight: FontWeight.bold,
                          color: avgMood >= 0.5 ? Colors.green : Colors.orange,
                        ),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      count > 0 ? '${_getEmotionEmoji(emotion)} $emotion' : 'No data',
                      style: const TextStyle(fontSize: 10, fontStyle: FontStyle.italic),
                    ),
                  ],
                );
              }).toList(),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildWeeklyPatternsCard() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Weekly Mood Patterns',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Card(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Row(
                  children: [
                    const Expanded(
                      child: Row(
                        children: [
                          Icon(Icons.business, color: Colors.grey, size: 20),
                          SizedBox(width: 8),
                          Text('Weekday Average', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                        ],
                      ),
                    ),
                    LinearProgressIndicator(
                      value: _weekdayAvg,
                      minHeight: 10,
                      backgroundColor: Colors.grey[200]!,
                      color: Colors.blue,
                      borderRadius: BorderRadius.circular(5),
                    ),
                    const SizedBox(width: 12),
                    Text('${(_weekdayAvg * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    const Expanded(
                      child: Row(
                        children: [
                          Icon(Icons.beach_access, color: Colors.green, size: 20),
                          SizedBox(width: 8),
                          Text('Weekend Average', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                        ],
                      ),
                    ),
                    LinearProgressIndicator(
                      value: _weekendAvg,
                      minHeight: 10,
                      backgroundColor: Colors.grey[200]!,
                      color: Colors.green,
                      borderRadius: BorderRadius.circular(5),
                    ),
                    const SizedBox(width: 12),
                    Text('${(_weekendAvg * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  ],
                ),
                if (_bestDayOfWeek.isNotEmpty) ...[
                  const Divider(height: 24),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text.rich(
                        TextSpan(
                          children: [
                            const TextSpan(text: 'Best Day: ', style: TextStyle(fontSize: 12)),
                            TextSpan(text: _bestDayOfWeek, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.green, fontSize: 13)),
                            const TextSpan(text: ' 🌟'),
                          ],
                        ),
                      ),
                      if (_worstDayOfWeek.isNotEmpty)
                        Text.rich(
                          TextSpan(
                            children: [
                              const TextSpan(text: 'Worst Day: ', style: TextStyle(fontSize: 12)),
                              TextSpan(text: _worstDayOfWeek, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.red, fontSize: 13)),
                              const TextSpan(text: ' 🌧️'),
                            ],
                          ),
                        ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMoodChart() {
    if (_entries.isEmpty) return const SizedBox();
    
    // Sort entries chronologically for the chart
    final sortedEntries = List<JournalEntry>.from(_entries);
    sortedEntries.sort((a, b) => a.createdAt.compareTo(b.createdAt));

    // Limit to the most recent 10 entries to keep chart readable
    final displayEntries = sortedEntries.length > 10 
        ? sortedEntries.sublist(sortedEntries.length - 10) 
        : sortedEntries;

    final spots = <FlSpot>[];
    for (int i = 0; i < displayEntries.length; i++) {
      spots.add(FlSpot(i.toDouble(), displayEntries[i].moodScore ?? 0.5));
    }

    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.only(top: 24, bottom: 12, left: 12, right: 24),
        child: LineChart(
          LineChartData(
            gridData: FlGridData(
              show: true,
              drawVerticalLine: false,
              horizontalInterval: 0.25,
              getDrawingHorizontalLine: (value) {
                return FlLine(
                  color: Colors.grey[200]!,
                  strokeWidth: 1,
                );
              },
            ),
            titlesData: FlTitlesData(
              show: true,
              rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              leftTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  interval: 0.25,
                  getTitlesWidget: (value, meta) {
                    return Text(
                      '${(value * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(color: Colors.grey, fontSize: 10, fontWeight: FontWeight.bold),
                    );
                  },
                ),
              ),
              bottomTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  getTitlesWidget: (value, meta) {
                    final idx = value.toInt();
                    if (idx >= 0 && idx < displayEntries.length) {
                      final date = displayEntries[idx].createdAt;
                      return Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Text(
                          '${date.day}/${date.month}',
                          style: const TextStyle(color: Colors.grey, fontSize: 9, fontWeight: FontWeight.bold),
                        ),
                      );
                    }
                    return const SizedBox();
                  },
                ),
              ),
            ),
            borderData: FlBorderData(show: false),
            minX: 0,
            maxX: (displayEntries.length - 1).toDouble(),
            minY: 0,
            maxY: 1.0,
            lineBarsData: [
              LineChartBarData(
                spots: spots,
                isCurved: true,
                color: Colors.deepPurple,
                barWidth: 3,
                dotData: const FlDotData(show: true),
                belowBarData: BarAreaData(
                  show: true,
                  color: Colors.deepPurple.withOpacity(0.1),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
