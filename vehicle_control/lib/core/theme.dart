import 'package:flutter/material.dart';

final ThemeData appTheme = ThemeData.dark().copyWith(
  primaryColor: Colors.blue,
  scaffoldBackgroundColor: Colors.black,
  textTheme: const TextTheme(
    bodyLarge: TextStyle(color: Colors.white),
    bodyMedium: TextStyle(color: Colors.white70),
  ),
);
