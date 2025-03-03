class SensorData {
  final double temperature;
  final double humidity;
  final double aqi;

  SensorData({required this.temperature, required this.humidity, required this.aqi});

  factory SensorData.fromJson(Map<String, dynamic> json) {
    return SensorData(
      temperature: json['temperature'].toDouble(),
      humidity: json['humidity'].toDouble(),
      aqi: json['aqi'].toDouble(),
    );
  }
}
