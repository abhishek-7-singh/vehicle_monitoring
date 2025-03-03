
import 'package:flutter/material.dart';
import 'screens/auth/login_screen.dart';
import 'screens/auth/signup_screen.dart';
import 'screens/dashboard/dashboard_screen.dart';
import 'screens/dashboard/charts_screen.dart';
import 'screens/data/real_time_sensor_screen.dart';
import 'screens/data/predictive_maintenance_screen.dart';
import 'screens/settings/profile_screen.dart';
import 'screens/settings/settings_screen.dart';
import 'screens/alerts/anomaly_detection_screen.dart';
import 'screens/alerts/notifications_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IoT Monitoring System',
      theme: ThemeData.dark(), // Dark theme for modern UI
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (context) => LoginScreen(),
        '/signup': (context) => SignupScreen(),
        '/dashboard': (context) => DashboardScreen(),
        '/charts': (context) => ChartsScreen(),
        '/real-time-sensors': (context) => RealTimeSensorScreen(),
        '/predictive-maintenance': (context) => PredictiveMaintenanceScreen(),
        '/profile': (context) => ProfileScreen(),
        '/settings': (context) => SettingsScreen(),
        '/alerts': (context) => AnomalyDetectionScreen(),
        '/notifications': (context) => NotificationsScreen(),
      },
    );
  }
}
