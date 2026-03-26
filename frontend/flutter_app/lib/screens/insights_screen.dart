import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class InsightsScreen extends StatefulWidget {
  const InsightsScreen({super.key});

  @override
  State<InsightsScreen> createState() => _InsightsScreenState();
}

class _InsightsScreenState extends State<InsightsScreen> {
  bool _isLoading = true;
  int _totalEntries = 0;
  double _averageMood = 0.0;
  String _moodTrend = 'No data';

  @override
  void initState() {
    super.initState();
    _loadInsights();
  }

  Future<void> _loadInsights() async {
    // TODO: Call API to load insights
    await Future.delayed(const Duration(seconds: 1));

    setState(() {
      _totalEntries = 0;
      _averageMood = 0.0;
      _moodTrend = 'No data';
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildStatCard(
            'Total Entries',
            _totalEntries.toString(),
            Icons.book,
            Colors.blue,
          ),
          const SizedBox(height: 12),
          _buildStatCard(
            'Average Mood',
            _averageMood > 0 ? _averageMood.toStringAsFixed(1) : '-',
            Icons.sentiment_satisfied,
            Colors.green,
          ),
          const SizedBox(height: 12),
          _buildStatCard(
            'Mood Trend',
            _moodTrend,
            Icons.trending_up,
            Colors.orange,
          ),
          const SizedBox(height: 24),
          if (_totalEntries > 0) ...[
            const Text(
              'Mood Over Time',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            SizedBox(height: 200, child: _buildMoodChart()),
          ] else ...[
            const Center(
              child: Column(
                children: [
                  SizedBox(height: 48),
                  Icon(Icons.insights, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'No insights yet',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  ),
                  SizedBox(height: 8),
                  Text(
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
    );
  }

  Widget _buildStatCard(
    String title,
    String value,
    IconData icon,
    Color color,
  ) {
    return Card(
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
              child: Icon(icon, color: color, size: 32),
            ),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontSize: 14, color: Colors.grey),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMoodChart() {
    // Sample data - will be replaced with real data from API
    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: false),
        titlesData: const FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: const [
              FlSpot(0, 0.5),
              FlSpot(1, 0.7),
              FlSpot(2, 0.6),
              FlSpot(3, 0.8),
              FlSpot(4, 0.9),
            ],
            isCurved: true,
            color: Colors.deepPurple,
            barWidth: 3,
            dotData: const FlDotData(show: true),
          ),
        ],
      ),
    );
  }
}
