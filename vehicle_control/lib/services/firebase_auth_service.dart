// import 'package:firebase_auth/firebase_auth.dart';
//
// class FirebaseAuthService {
//   final FirebaseAuth _auth = FirebaseAuth.instance;
//
//   Future<User?> signInWithEmail(String email, String password) async {
//     try {
//       UserCredential userCredential =
//       await _auth.signInWithEmailAndPassword(email: email, password: password);
//       return userCredential.user;
//     } catch (e) {
//       print('Error: $e');
//       return null;
//     }
//   }
//
//   Future<User?> signUpWithEmail(String email, String password) async {
//     try {
//       UserCredential userCredential =
//       await _auth.createUserWithEmailAndPassword(email: email, password: password);
//       return userCredential.user;
//     } catch (e) {
//       print('Error: $e');
//       return null;
//     }
//   }
// }
