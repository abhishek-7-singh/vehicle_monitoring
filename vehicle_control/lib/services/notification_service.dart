// import 'package:flutter_local_notifications/flutter_local_notifications.dart';
//
// class NotificationService {
//   final FlutterLocalNotificationsPlugin _notificationsPlugin =
//   FlutterLocalNotificationsPlugin();
//
//   Future<void> initNotifications() async {
//     var androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
//     var initSettings = InitializationSettings(android: androidSettings);
//
//     await _notificationsPlugin.initialize(initSettings);
//   }
//
//   Future<void> showNotification(String title, String body) async {
//     var androidDetails = AndroidNotificationDetails('channelId', 'channelName',
//         importance: Importance.high);
//     var notificationDetails = NotificationDetails(android: androidDetails);
//
//     await _notificationsPlugin.show(0, title, body, notificationDetails);
//   }
// }
