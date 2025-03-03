import 'package:flutter/material.dart';

class AnomalyDetectionScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Anomaly Detection'),
        backgroundColor: Colors.redAccent,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.warning_amber_rounded, size: 100, color: Colors.red),
            SizedBox(height: 20),
            Text(
              'No Anomalies Detected',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: () {
                // Add logic to check anomalies
              },
              child: Text('Refresh'),
            ),
          ],
        ),
      ),
    );
  }
}
