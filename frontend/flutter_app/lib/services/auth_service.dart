import 'dart:convert';
import 'api_service.dart';
import '../models/user.dart';

class AuthService {
  static Future<User> register(
    String email,
    String username,
    String password,
  ) async {
    final response = await ApiService.post('/auth/register', {
      'email': email,
      'username': username,
      'password': password,
    });

    if (response.statusCode == 200) {
      return User.fromJson(jsonDecode(response.body));
    } else {
      throw Exception(
        jsonDecode(response.body)['detail'] ?? 'Registration failed',
      );
    }
  }

  static Future<String> login(String username, String password) async {
    final response = await ApiService.post('/auth/login', {
      'username': username,
      'password': password,
    });

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      await ApiService.saveToken(data['access_token']);
      return data['access_token'];
    } else {
      throw Exception('Invalid credentials');
    }
  }

  static Future<void> logout() async {
    await ApiService.clearToken();
  }

  static Future<bool> isLoggedIn() async {
    final token = await ApiService.getToken();
    return token != null;
  }

  static Future<String?> getToken() async {
    return await ApiService.getToken();
  }
}
