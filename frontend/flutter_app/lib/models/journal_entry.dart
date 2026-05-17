class JournalEntry {
  final int id;
  final int userId;
  final String content;
  final double? moodScore;
  final String? moodCategory;
  final String? sentimentLabel;
  final String? dominantTopic;
  final Map<String, dynamic>? emotionDistribution;
  final Map<String, dynamic>? topicsDistribution;
  final String? combinedInsight;
  final String analysisStatus;
  final DateTime createdAt;
  final DateTime updatedAt;

  JournalEntry({
    required this.id,
    required this.userId,
    required this.content,
    this.moodScore,
    this.moodCategory,
    this.sentimentLabel,
    this.dominantTopic,
    this.emotionDistribution,
    this.topicsDistribution,
    this.combinedInsight,
    required this.analysisStatus,
    required this.createdAt,
    required this.updatedAt,
  });

  factory JournalEntry.fromJson(Map<String, dynamic> json) {
    return JournalEntry(
      id: json['id'],
      userId: json['user_id'],
      content: json['content'],
      moodScore: json['mood_score']?.toDouble(),
      moodCategory: json['mood_category'],
      sentimentLabel: json['sentiment_label'],
      dominantTopic: json['dominant_topic'],
      emotionDistribution: json['emotion_distribution'] != null 
          ? Map<String, dynamic>.from(json['emotion_distribution']) 
          : null,
      topicsDistribution: json['topics_distribution'] != null 
          ? Map<String, dynamic>.from(json['topics_distribution']) 
          : null,
      combinedInsight: json['combined_insight'],
      analysisStatus: json['analysis_status'] ?? 'pending',
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {'content': content};
  }
}
