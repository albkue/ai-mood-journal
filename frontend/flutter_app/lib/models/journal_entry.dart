class JournalEntry {
  final int id;
  final int userId;
  final String content;
  final double? moodScore;
  final String? sentimentLabel;
  final DateTime createdAt;
  final DateTime updatedAt;

  JournalEntry({
    required this.id,
    required this.userId,
    required this.content,
    this.moodScore,
    this.sentimentLabel,
    required this.createdAt,
    required this.updatedAt,
  });

  factory JournalEntry.fromJson(Map<String, dynamic> json) {
    return JournalEntry(
      id: json['id'],
      userId: json['user_id'],
      content: json['content'],
      moodScore: json['mood_score']?.toDouble(),
      sentimentLabel: json['sentiment_label'],
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {'content': content};
  }
}
