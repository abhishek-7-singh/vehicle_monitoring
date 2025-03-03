// import 'package:flutter/material.dart';
//
// class ChartsScreen extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Charts')),
//       body: Center(child: Text('Charts and analytics will be displayed here')),
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class ChartsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Vehicle Analytics')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Vehicle Speed Over Time',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 200, child: LineChart(_lineChartData())),
            SizedBox(height: 20),
            Text(
              'Fuel Efficiency by Month',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 200, child: BarChart(_barChartData())),
            SizedBox(height: 20),
            Text(
              'Engine Performance Breakdown',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 200, child: PieChart(_pieChartData())),
          ],
        ),
      ),
    );
  }

  LineChartData _lineChartData() {
    return LineChartData(
      gridData: FlGridData(show: false),
      titlesData: FlTitlesData(show: false),
      borderData: FlBorderData(show: false),
      lineBarsData: [
        LineChartBarData(
          spots: [FlSpot(0, 20), FlSpot(1, 30), FlSpot(2, 50), FlSpot(3, 40), FlSpot(4, 60)],
          isCurved: true,
          color: Colors.blue, // Updated from 'colors' to 'color'
          dotData: FlDotData(show: false),
        ),
      ],
    );
  }

  BarChartData _barChartData() {
    return BarChartData(
      barGroups: [
        BarChartGroupData(x: 0, barRods: [BarChartRodData(toY: 12, color: Colors.orange)]),
        BarChartGroupData(x: 1, barRods: [BarChartRodData(toY: 15, color: Colors.orange)]),
        BarChartGroupData(x: 2, barRods: [BarChartRodData(toY: 18, color: Colors.orange)]),
      ],
    );
  }

  PieChartData _pieChartData() {
    return PieChartData(
      sections: [
        PieChartSectionData(value: 40, color: Colors.green, title: 'Good'),
        PieChartSectionData(value: 30, color: Colors.yellow, title: 'Average'),
        PieChartSectionData(value: 30, color: Colors.red, title: 'Poor'),
      ],
    );
  }
}
