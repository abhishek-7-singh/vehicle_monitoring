import 'package:flutter/material.dart';

class RealTimeSensorScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Real-time Sensors')),
      body: Center(child: Text('Live sensor data will be displayed here')),
    );
  }
}
