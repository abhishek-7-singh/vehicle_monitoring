// import 'package:flutter/material.dart';
// import 'charts_screen.dart';
//
// class DashboardScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Dashboard')),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(context, MaterialPageRoute(builder: (context) => ChartsScreen()));
//               },
//               child: Text('View Charts'),
//             ),
//             SizedBox(height: 20),
//             Text('Real-time vehicle data will appear here'),
//           ],
//         ),
//       ),
//     );
//   }
// }


// import 'package:flutter/material.dart';
// import 'charts_screen.dart';
// import '../data/real_time_sensor_screen.dart';
// import '../data/predictive_maintenance_screen.dart';
// import '../settings/profile_screen.dart';
// import '../settings/settings_screen.dart';
// import '../alerts/anomaly_detection_screen.dart';
// import '../alerts/notifications_screen.dart';
//
// class DashboardScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Dashboard')),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => ChartsScreen()));
//               },
//               child: Text('View Charts'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => RealTimeSensorScreen()));
//               },
//               child: Text('Real-Time Sensors'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => PredictiveMaintenanceScreen()));
//               },
//               child: Text('Predictive Maintenance'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => ProfileScreen()));
//               },
//               child: Text('Profile'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => SettingsScreen()));
//               },
//               child: Text('Settings'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => AnomalyDetectionScreen()));
//               },
//               child: Text('Anomaly Detection'),
//             ),
//             SizedBox(height: 10),
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.push(
//                     context, MaterialPageRoute(builder: (context) => NotificationsScreen()));
//               },
//               child: Text('Notifications'),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
//


// import 'package:flutter/material.dart';
// import 'charts_screen.dart';
// import '../data/real_time_sensor_screen.dart';
// import '../data/predictive_maintenance_screen.dart';
// import '../settings/profile_screen.dart';
// import '../settings/settings_screen.dart';
// import '../alerts/anomaly_detection_screen.dart';
// import '../alerts/notifications_screen.dart';
//
// class DashboardScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Dashboard'),
//         actions: [
//           PopupMenuButton<String>(
//             onSelected: (String route) {
//               Navigator.pushNamed(context, route);
//             },
//             itemBuilder: (BuildContext context) => [
//               _buildMenuItem('Charts', '/charts'),
//               _buildMenuItem('Real-Time Sensors', '/real-time-sensors'),
//               _buildMenuItem('Predictive Maintenance', '/predictive-maintenance'),
//               _buildMenuItem('Profile', '/profile'),
//               _buildMenuItem('Settings', '/settings'),
//               _buildMenuItem('Anomaly Detection', '/alerts'),
//               _buildMenuItem('Notifications', '/notifications'),
//             ],
//           ),
//         ],
//       ),
//       body: Center(
//         child: Text('Welcome to the Dashboard! Tap the three dots (⋮) in the top-right for the menu.'),
//       ),
//     );
//   }
//
//   PopupMenuItem<String> _buildMenuItem(String title, String route) {
//     return PopupMenuItem<String>(
//       value: route,
//       child: Text(title),
//     );
//   }
// }


// import 'package:flutter/material.dart';
// import 'charts_screen.dart';
// import '../data/real_time_sensor_screen.dart';
// import '../data/predictive_maintenance_screen.dart';
// import '../settings/profile_screen.dart';
// import '../settings/settings_screen.dart';
// import '../alerts/anomaly_detection_screen.dart';
// import '../alerts/notifications_screen.dart';
//
// class DashboardScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Dashboard'),
//       ),
//       drawer: Drawer(
//         child: ListView(
//           padding: EdgeInsets.zero,
//           children: [
//             DrawerHeader(
//               decoration: BoxDecoration(color: Colors.blue),
//               child: Text('Menu', style: TextStyle(color: Colors.white, fontSize: 24)),
//             ),
//             _buildDrawerItem(context, 'Charts', '/charts'),
//             _buildDrawerItem(context, 'Real-Time Sensors', '/real-time-sensors'),
//             _buildDrawerItem(context, 'Predictive Maintenance', '/predictive-maintenance'),
//             _buildDrawerItem(context, 'Profile', '/profile'),
//             _buildDrawerItem(context, 'Settings', '/settings'),
//             _buildDrawerItem(context, 'Anomaly Detection', '/alerts'),
//             _buildDrawerItem(context, 'Notifications', '/notifications'),
//           ],
//         ),
//       ),
//       body: Center(
//         child: Text('Welcome to the Dashboard! Swipe from the left or tap ☰ to open the menu.'),
//       ),
//     );
//   }
//
//   Widget _buildDrawerItem(BuildContext context, String title, String route) {
//     return ListTile(
//       title: Text(title),
//       onTap: () {
//         Navigator.pop(context); // Close drawer
//         Navigator.pushNamed(context, route);
//       },
//     );
//   }
// }
// import 'package:flutter/material.dart';
// import 'charts_screen.dart';
// import '../data/real_time_sensor_screen.dart';
// import '../data/predictive_maintenance_screen.dart';
// import '../settings/profile_screen.dart';
// import '../settings/settings_screen.dart';
// import '../alerts/anomaly_detection_screen.dart';
// import '../alerts/notifications_screen.dart';
// import 'package:font_awesome_flutter/font_awesome_flutter.dart';
//
// class DashboardScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Dashboard', style: TextStyle(fontWeight: FontWeight.bold)),
//         centerTitle: true,
//       ),
//       drawer: _buildDrawer(context),
//       body: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Column(
//           crossAxisAlignment: CrossAxisAlignment.start,
//           children: [
//             Text('Overview', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
//             SizedBox(height: 10),
//             _buildDashboardGrid(),
//           ],
//         ),
//       ),
//     );
//   }
//
//   Widget _buildDrawer(BuildContext context) {
//     return Drawer(
//       child: ListView(
//         padding: EdgeInsets.zero,
//         children: [
//           DrawerHeader(
//             decoration: BoxDecoration(color: Colors.blue),
//             child: Text('Menu', style: TextStyle(color: Colors.white, fontSize: 24)),
//           ),
//           _buildDrawerItem(context, 'Charts', '/charts', Icons.bar_chart),
//           _buildDrawerItem(context, 'Real-Time Sensors', '/real-time-sensors', Icons.sensors),
//           _buildDrawerItem(context, 'Predictive Maintenance', '/predictive-maintenance', Icons.settings_suggest),
//           _buildDrawerItem(context, 'Profile', '/profile', Icons.person),
//           _buildDrawerItem(context, 'Settings', '/settings', Icons.settings),
//           _buildDrawerItem(context, 'Anomaly Detection', '/alerts', Icons.warning),
//           _buildDrawerItem(context, 'Notifications', '/notifications', Icons.notifications),
//         ],
//       ),
//     );
//   }
//
//   Widget _buildDrawerItem(BuildContext context, String title, String route, IconData icon) {
//     return ListTile(
//       leading: Icon(icon, color: Colors.blue),
//       title: Text(title),
//       onTap: () {
//         Navigator.pop(context);
//         Navigator.pushNamed(context, route);
//       },
//     );
//   }
//
//   Widget _buildDashboardGrid() {
//     return Expanded(
//       child: GridView.count(
//         crossAxisCount: 2,
//         crossAxisSpacing: 10,
//         mainAxisSpacing: 10,
//         children: [
//           _buildDashboardCard('Services Done', '12', FontAwesomeIcons.tools, Colors.green),
//           _buildDashboardCard('Vehicle Number', 'AB-1234', FontAwesomeIcons.car, Colors.blue),
//           _buildDashboardCard('Licenses', 'Valid', FontAwesomeIcons.idCard, Colors.orange),
//           _buildDashboardCard('Engine Health', 'Good', FontAwesomeIcons.oilCan, Colors.red),
//         ],
//       ),
//     );
//   }
//
//   Widget _buildDashboardCard(String title, String value, IconData icon, Color color) {
//     return Card(
//       shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
//       elevation: 4,
//       child: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Icon(icon, size: 40, color: color),
//             SizedBox(height: 10),
//             Text(title, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
//             SizedBox(height: 5),
//             Text(value, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black54)),
//           ],
//         ),
//       ),
//     );
//   }
// }


import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Dashboard'),
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.blue),
              child: Text('Menu', style: TextStyle(color: Colors.white, fontSize: 24)),
            ),
            _buildDrawerItem(context, 'Charts', '/charts'),
            _buildDrawerItem(context, 'Real-Time Sensors', '/real-time-sensors'),
            _buildDrawerItem(context, 'Predictive Maintenance', '/predictive-maintenance'),
            _buildDrawerItem(context, 'Profile', '/profile'),
            _buildDrawerItem(context, 'Settings', '/settings'),
            _buildDrawerItem(context, 'Anomaly Detection', '/alerts'),
            _buildDrawerItem(context, 'Notifications', '/notifications'),
          ],
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: GridView.count(
                crossAxisCount: 2,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                children: [
                  _buildDashboardCard('Services Done', '12', FontAwesomeIcons.wrench, Colors.blue),
                  _buildDashboardCard('Vehicle Number', 'AB 1234 XY', FontAwesomeIcons.car, Colors.orange),
                  _buildDashboardCard('Licenses', 'Valid', FontAwesomeIcons.idCard, Colors.green),
                  _buildDashboardCard('Engine Health', 'Good', FontAwesomeIcons.cogs, Colors.red),
                ],
              ),
            ),
            SizedBox(height: 16),
            _buildLargeDashboardCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildDashboardCard(String title, String value, IconData icon, Color color) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 40, color: color),
            SizedBox(height: 10),
            Text(title, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            SizedBox(height: 5),
            Text(value, style: TextStyle(fontSize: 14, color: Colors.grey[700])),
          ],
        ),
      ),
    );
  }

  Widget _buildLargeDashboardCard() {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 5,
      color: Colors.grey[100],
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Vehicle Performance Overview',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold,color: Colors.blue[700]),
            ),
            Divider(thickness: 1),
            SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatItem('Total Km', '120,450', FontAwesomeIcons.road, Colors.blue,),
                _buildStatItem('Avg Speed', '65 km/h', FontAwesomeIcons.tachometerAlt, Colors.green),
                _buildStatItem('Avg Gear', '4', FontAwesomeIcons.cogs, Colors.orange),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(String title, String value, IconData icon, Color color) {
    return Column(
      children: [
        Icon(icon, size: 30, color: color),
        SizedBox(height: 5),
        Text(title, style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold,color: Colors.blue[700])),
        SizedBox(height: 3),
        Text(value, style: TextStyle(fontSize: 14, color: Colors.blue[700])),
      ],
    );
  }

  Widget _buildDrawerItem(BuildContext context, String title, String route) {
    return ListTile(
      title: Text(title),
      onTap: () {
        Navigator.pop(context); // Close drawer
        Navigator.pushNamed(context, route);
      },
    );
  }
}
