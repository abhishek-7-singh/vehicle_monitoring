#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#define RX_PIN 3  // Recommended RX pin for ESP32-CAM
#define TX_PIN 1  // Recommended TX pin for ESP32-CAM

// WiFi Credentials
const char* ssid = "sri";
const char* password = "1234567890";

// Server URL
const char* server_url = "https://engine-carbon-deposition-prediction.onrender.com";  // Change this to your server IP

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);  // Using Serial1

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(1000);
    }
    HTTPClient http;
    http.setTimeout(5000);  
    http.begin(String(server_url) + "/device_status");
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<128> doc;
    doc["message"] = "Iam Here!!!";  // Wrap the message in a JSON object

    String jsonStr;
    serializeJson(doc, jsonStr);

    int httpResponseCode = http.POST(jsonStr);  // Use POST instead of GET

    Serial.println("Response Code: " + String(httpResponseCode));

    http.end();
    Serial.println("\nWiFi Connected!");
}

void loop() {
    if (Serial1.available()) {
        String inputData = Serial1.readStringUntil('\n');  // Read JSON from Serial
        Serial.println("Received JSON: " + inputData);

        DynamicJsonDocument doc(2048);
        DeserializationError error = deserializeJson(doc, inputData);

        if (error) {
            Serial.println("JSON Parsing Failed!");
            return;
        }

        if (doc.containsKey("ignition")) {
            sendIgnitionStatus(doc["ignition"]);
        } else if (doc.is<JsonObject>()) {
            sendRealTimeData(doc.as<JsonObject>());  // Send sensor data
        }
    }
    delay(500);
}

// 📌 Function to send real-time data to /upload-realtime
void sendRealTimeData(JsonObject obj) {
    HTTPClient http;
    http.setTimeout(5000);  
    http.begin(String(server_url) + "/upload-realtime");
    http.addHeader("Content-Type", "application/json");

    String jsonStr;
    serializeJson(obj, jsonStr);
    Serial.println("Sending Real-Time Data: " + jsonStr);

    int httpResponseCode = http.POST(jsonStr);
    Serial.println("Response: " + String(httpResponseCode));

    http.end();
}

// 📌 Function to send ignition status separately to /update-ignition
void sendIgnitionStatus(bool ignitionStatus) {
    HTTPClient http;
    http.setTimeout(5000);  
    http.begin(String(server_url) + "/update-ignition");
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<128> doc;
    doc["ignition"] = ignitionStatus;

    String jsonStr;
    serializeJson(doc, jsonStr);
    Serial.println("Sending Ignition Status: " + jsonStr);

    int httpResponseCode = http.POST(jsonStr);
    Serial.println("Response: " + String(httpResponseCode));

    http.end();
}

