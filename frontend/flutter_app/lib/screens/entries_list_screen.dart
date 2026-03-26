import 'package:flutter/material.dart';
import '../models/journal_entry.dart';

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
    // TODO: Call API to load entries
    await Future.delayed(const Duration(seconds: 1));

    setState(() {
      _entries = []; // Will be populated from API
      _isLoading = false;
    });
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_entries.isEmpty) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.book, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              'No entries yet',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
            SizedBox(height: 8),
            Text(
              'Start writing your first journal entry!',
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _entries.length,
      itemBuilder: (context, index) {
        final entry = _entries[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            title: Text(
              entry.content.length > 50
                  ? '${entry.content.substring(0, 50)}...'
                  : entry.content,
            ),
            subtitle: Text(_formatDate(entry.createdAt)),
            trailing: entry.moodScore != null
                ? Chip(
                    label: Text('${entry.moodScore!.toStringAsFixed(1)}'),
                    backgroundColor: entry.moodScore! > 0.5
                        ? Colors.green[100]
                        : Colors.orange[100],
                  )
                : null,
            onTap: () {
              // TODO: Navigate to entry detail
            },
          ),
        );
      },
    );
  }
}
