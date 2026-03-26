class User {
  final int id;
  final String email;
  final String username;
  final DateTime createdAt;

  User({
    required this.id,
    required this.email,
    required this.username,
    required this.createdAt,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      email: json['email'],
      username: json['username'],
      createdAt: DateTime.parse(json['created_at']),
    );
  }
}
