// import 'package:firebase_database/firebase_database.dart';
//
// class FirebaseDatabaseService {
//   final DatabaseReference _database = FirebaseDatabase.instance.reference();
//
//   Future<void> saveSensorData(String path, Map<String, dynamic> data) async {
//     await _database.child(path).set(data);
//   }
//
//   DatabaseReference getSensorData(String path) {
//     return _database.child(path);
//   }
// }
