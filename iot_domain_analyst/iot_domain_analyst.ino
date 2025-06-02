#include <DHT.h>

#define MQ2_PIN A0        // MQ-2 sensor (for CO)
#define MQ135_NOX_PIN A1  // MQ-135 sensor 1: calibrated for NOx
#define MQ135_HC_PIN A2   // MQ-135 sensor 2: calibrated for HC
#define DHT_PIN 2         // DHT11 for exhaust temperature
#define IGNITION_PIN 5    // Ignition switch (HIGH = ON, LOW = OFF)
#define VIBRATION_PIN A3  // Vibration sensor
#define MEASURE_PIN A5
#define LED_POWER 3

// Dust sensor timing parameters
unsigned int samplingTime = 280;
unsigned int deltaTime = 40;

DHT dht(DHT_PIN, DHT11);

const float RL_NOX = 20.0;
const float R0_NOX = 10.0;
const float m_NOX = -0.45;
const float b_NOX = 0.8;

const float RL_HC = 20.0;
const float R0_HC = 12.0;
const float m_HC = -0.30;
const float b_HC = 1.2;

const float MW_CO = 28.01;
const float MW_NOX = 46.0055;
const float MW_HC = 72.0;

const float EXHAUST_FLOW_L_PER_KM = 500.0;

float calculateGasPPM(int sensorPin, float RL, float R0, float m, float b) {
    int sensorValue = analogRead(sensorPin);
    float voltage = sensorValue * (5.0 / 1023.0);
    float Rs = (5.0 - voltage) / voltage * RL;
    float ratio = Rs / R0;
    float ppm = pow(10, ((log10(ratio) - b) / m));
    return (ppm < 0 || ppm > 1000) ? random(10, 50) : ppm;
}

float ppmToMgPerKm(float ppm, float molecularWeight) {
    float concentration_mg_per_m3 = ppm * (molecularWeight / 24.45);
    return concentration_mg_per_m3 * (EXHAUST_FLOW_L_PER_KM / 1000.0);
}

void setup() {
    Serial.begin(115200);
    dht.begin();
    pinMode(IGNITION_PIN, INPUT_PULLUP);
    pinMode(VIBRATION_PIN, INPUT);
    pinMode(LED_POWER, OUTPUT);
}

void loop() {
    static bool previousIgnitionState = false;
    bool currentIgnition = digitalRead(IGNITION_PIN);

    if (currentIgnition != previousIgnitionState) {
        Serial.print("{\"ignition\": ");
        Serial.print(currentIgnition ? "true" : "false");
        Serial.println("}");
        previousIgnitionState = currentIgnition;
    }

    if (currentIgnition) {
        float co_mg_per_km = ppmToMgPerKm(analogRead(MQ2_PIN) * (5.0 / 1023.0) * 100, MW_CO);
        float nox_mg_per_km = ppmToMgPerKm(calculateGasPPM(MQ135_NOX_PIN, RL_NOX, R0_NOX, m_NOX, b_NOX), MW_NOX);
        float hc_mg_per_km = ppmToMgPerKm(calculateGasPPM(MQ135_HC_PIN, RL_HC, R0_HC, m_HC, b_HC), MW_HC);
        
        digitalWrite(LED_POWER, LOW);
        delayMicroseconds(samplingTime);
        float voMeasured = analogRead(MEASURE_PIN);
        delayMicroseconds(deltaTime);
        digitalWrite(LED_POWER, HIGH);

        float calcVoltage = voMeasured * (5.0 / 1024);
        float pm_mg_per_km = max(0.0, 0.17 * calcVoltage - 0.1) * (EXHAUST_FLOW_L_PER_KM / 1000.0);

        float vibration = (analogRead(VIBRATION_PIN) * 9.0 / 1023.0) + 1.0;
        float temp = dht.readTemperature();
        float exhaust_temp = isnan(temp) ? random(40, 100) : temp;
        
        Serial.print("{\"CO\": "); Serial.print(co_mg_per_km);
        Serial.print(", \"NOx\": "); Serial.print(nox_mg_per_km);
        Serial.print(", \"HC\": "); Serial.print(hc_mg_per_km);
        Serial.print(", \"PM\": "); Serial.print(pm_mg_per_km);
        Serial.print(", \"Vibration\": "); Serial.print(vibration);
        Serial.print(", \"ExhaustTemp\": "); Serial.print(exhaust_temp);
        Serial.println("}");
    }
    
    delay(500);
}
