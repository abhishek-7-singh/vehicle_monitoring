require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const admin = require('firebase-admin');
const cors = require('cors');
const fs = require('fs');

// Initialize Firebase Admin SDK
const serviceAccount = JSON.parse(fs.readFileSync('serviceAccountKey.json', 'utf8'));

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: process.env.FIREBASE_DB_URL  // Set this in `.env`
});

const db = admin.firestore();
const realtimeDB = admin.database();  // 🔥 Realtime Database

const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 5000;

// 🚀 **Route to Upload Sensor Data**
app.post('/upload-realtime', async (req, res) => {
    try {
        const data = req.body;
        console.log("📥 Sensor Data Received:", data);

        if (!data || typeof data !== 'object') {
            return res.status(400).json({ error: "Invalid data format. Expected an object." });
        }

        // **🔥 1. Update Realtime Database**
        const liveRef = realtimeDB.ref("live_sensor_data");
        await liveRef.update({
            CO: data.CO,
            NOx: data.NOx,
            HC_NOx: data.HC,
            PM: data.PM,
            Vibration: data.Vibration,
            ExhaustTemp: data.ExhaustTemp,
            timestamp: admin.database.ServerValue.TIMESTAMP
        });

        // **🔥 2. Store Data in Firestore**
        const docRef = db.collection('vehicle_data').doc(); // Auto-generate doc ID
        await docRef.set({
            CO: data.CO,
            NOx: data.NOx,
            HC_NOx: data.HC,
            PM: data.PM,
            Vibration: data.Vibration,
            ExhaustTemp: data.ExhaustTemp,
            timestamp: admin.firestore.FieldValue.serverTimestamp()
        });

        res.json({ message: "✅ Sensor data uploaded successfully!" });

    } catch (error) {
        console.error("❌ Error uploading sensor data:", error);
        res.status(500).json({ error: "Internal server error" });
    }
});

// 🚀 **New Route to Handle Ignition Status**
app.post('/update-ignition', async (req, res) => {
    try {
        const { ignition } = req.body;
        console.log("📥 Ignition Status Received:", ignition);

        if (typeof ignition !== 'boolean') {
            return res.status(400).json({ error: "Invalid data format. Expected a boolean value." });
        }

        // **🔥 Update Realtime Database**
        const ignitionRef = realtimeDB.ref("vehicle_status");
        await ignitionRef.update({
            ignition: ignition,
            timestamp: admin.database.ServerValue.TIMESTAMP
        });

        // **🔥 Store in Firestore**
        await db.collection('vehicle_status').add({
            ignition: ignition,
            timestamp: admin.firestore.FieldValue.serverTimestamp()
        });

        res.json({ message: `✅ Ignition status updated: ${ignition}` });

    } catch (error) {
        console.error("❌ Error updating ignition status:", error);
        res.status(500).json({ error: "Internal server error" });
    }
});
app.post('/device_status', async (req, res) => {
    try {
        const status = req.body;
        console.log("✅ Device connected!!!!", status);

        res.json({ message: "Device status received successfully!" });  // Send a response
    } catch (error) {
        console.error("❌ Error:", error);
        res.status(500).json({ error: "Internal Server Error" });  // Send an error response
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});