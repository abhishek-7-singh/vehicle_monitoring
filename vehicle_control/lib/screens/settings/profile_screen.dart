// import 'package:flutter/material.dart';
//
// class ProfileScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Profile')),
//       body: Center(child: Text('User profile details')),
//     );
//   }
// }

// import 'package:flutter/material.dart';
//
// class ProfileScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Profile')),
//       body: Padding(
//         padding: EdgeInsets.all(16.0),
//         child: Column(
//           crossAxisAlignment: CrossAxisAlignment.center,
//           children: [
//             // Profile Picture
//             CircleAvatar(
//               radius: 60,
//               backgroundImage: AssetImage('assets/profile_pic.jpg'), // Add an image to assets
//             ),
//             SizedBox(height: 12),
//
//             // User Name
//             Text(
//               'John Doe',
//               style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
//             ),
//             SizedBox(height: 6),
//             Text('Vehicle Owner', style: TextStyle(fontSize: 16, color: Colors.grey)),
//
//             Divider(height: 30, thickness: 1),
//
//             // Vehicle Information
//             _buildInfoRow('Owner Name', 'John Doe'),
//             _buildInfoRow('Contact Number', '+91 98765 43210'),
//             _buildInfoRow('Vehicle Name', 'Tesla Model X'),
//             _buildInfoRow('Vehicle Number', 'MH 12 AB 3456'),
//             _buildInfoRow('Model Year', '2022'),
//             _buildInfoRow('Fuel Type', 'Electric'),
//             _buildInfoRow('Insurance Validity', '12 Dec 2025'),
//             _buildInfoRow('Registration State', 'Maharashtra'),
//
//             SizedBox(height: 20),
//
//             // Edit Profile Button
//             ElevatedButton(
//               onPressed: () {
//                 // Add functionality if needed
//               },
//               child: Text('Edit Profile'),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
//
//   Widget _buildInfoRow(String label, String value) {
//     return Padding(
//       padding: const EdgeInsets.symmetric(vertical: 6.0),
//       child: Row(
//         mainAxisAlignment: MainAxisAlignment.spaceBetween,
//         children: [
//           Text(label, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500)),
//           Text(value, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w400, color: Colors.white70)),
//         ],
//       ),
//     );
//   }
// }


import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('Profile', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blueGrey.shade900, Colors.black87],
          ),
        ),
        child: Column(
          children: [
            // Profile Picture & Name Section
            _buildProfileHeader(),

            // Details Section
            Expanded(
              child: SingleChildScrollView(
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      _buildInfoCard('Owner Name', 'Abhishek', FontAwesomeIcons.user),
                      _buildInfoCard('Contact Number', '+91 98765 43210', FontAwesomeIcons.phone),
                      _buildInfoCard('Vehicle Name', 'Tesla Model X', FontAwesomeIcons.car),
                      _buildInfoCard('Vehicle Number', 'MH 12 AB 3456', FontAwesomeIcons.idBadge),
                      _buildInfoCard('Model Year', '2022', FontAwesomeIcons.calendar),
                      _buildInfoCard('Fuel Type', 'Electric', FontAwesomeIcons.bolt),
                      _buildInfoCard('Insurance Validity', '12 Dec 2025', FontAwesomeIcons.shieldAlt),
                      _buildInfoCard('Registration State', 'Maharashtra', FontAwesomeIcons.mapMarkerAlt),

                      SizedBox(height: 20),

                      // Edit Profile Button
                      ElevatedButton.icon(
                        onPressed: () {},
                        icon: Icon(Icons.edit, color: Colors.white),
                        label: Text('Edit Profile'),
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                          backgroundColor: Colors.blueAccent.shade700,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Profile Header with Picture & Name
  Widget _buildProfileHeader() {
    return Container(
      alignment: Alignment.center,
      padding: EdgeInsets.symmetric(vertical: 40),
      child: Column(
        children: [
          CircleAvatar(
            radius: 60,
            backgroundImage: AssetImage(''), // Add profile image to assets
          ),
          SizedBox(height: 12),
          Text(
            'Abhishek Singh',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.white),
          ),
          Text(
            'Vehicle Owner',
            style: TextStyle(fontSize: 16, color: Colors.grey.shade300),
          ),
        ],
      ),
    );
  }

  // Information Card with Icons
  Widget _buildInfoCard(String title, String value, IconData icon) {
    return Card(
      color: Colors.blueGrey.shade800,
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        leading: Icon(icon, color: Colors.white70, size: 28),
        title: Text(title, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: Colors.white)),
        subtitle: Text(value, style: TextStyle(fontSize: 14, color: Colors.grey.shade300)),
      ),
    );
  }
}
