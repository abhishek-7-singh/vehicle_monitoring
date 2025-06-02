import firebase_admin
from firebase_admin import credentials, firestore, db
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
from sklearn.ensemble import IsolationForest


#  Firebase Setup

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://veh-pa-default-rtdb.firebaseio.com/'
})
db_firestore = firestore.client()
db_realtime = db.reference()
collection_name = "vehicle_data"


# Load Pre-Trained PyTorch LSTM Model & Scalers
class CarbonLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CarbonLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out[:, -1, :])  
        out = self.fc(out)
        return out

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CarbonLSTM(input_dim=6, hidden_dim=128, num_layers=2, output_dim=1).to(device)

# Handle state_dict key mismatch
state_dict = torch.load("carbon_lstm_model.pth", map_location=device)
new_state_dict = {k.replace("fc.", "fc.0.") if "fc." in k else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Load scalers
scaler_features = joblib.load("scaler_features.pkl")
scaler_target = joblib.load("scaler_target.pkl")


# Function to Fetch Latest Data from Firestore

def fetch_latest_data():
    docs = db_firestore.collection(collection_name).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
    data = [doc.to_dict() for doc in docs]
    if not data:
        print("No data available in Firestore.")
        return None
    df = pd.DataFrame(data).sort_values(by="timestamp")  # Ensure correct time order
    return df


#  Function to Predict Carbon Deposition Using LSTM

SEQ_LENGTH = 10

def create_sequences(data, seq_length):
    return np.array([data[i:i+seq_length, :] for i in range(len(data) - seq_length)])

def predict_future_values(df):
    features = ['CO', 'NOx', 'HC_NOx', 'PM', 'Vibration', 'ExhaustTemp']
    df_features = df[features]
    scaled_features = scaler_features.transform(df_features)
    if len(df_features) < SEQ_LENGTH:
        print("\n Not enough data for LSTM prediction!")
        return None
    X_input = create_sequences(scaled_features, SEQ_LENGTH)
    X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
    predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
    future_timestamps = pd.date_range(start=df["timestamp"].iloc[-1], periods=len(predictions)+1, freq="1H")[1:]
    forecast_df = pd.DataFrame({"timestamp": future_timestamps, "predicted_carbon_deposit": predictions})
    forecast_df.to_csv("forecast.csv", index=False)
    print("\n Forecast saved as 'forecast.csv'")
    return forecast_df


#  Function for Anomaly Detection

def detect_anomalies(df):
    if "Carbon Deposit Level" not in df.columns:
        print("\n 'Carbon Deposit Level' column not found!")
        return None
    df["Carbon Deposit Level"].fillna(df["Carbon Deposit Level"].mean(), inplace=True)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = iso_forest.fit_predict(df[["Carbon Deposit Level"]])
    anomalies = df[df["anomaly_score"] == -1]
    return anomalies


# Function to Trigger Maintenance Alert (Realtime Database)

def trigger_maintenance_alert(predictions, anomalies):
    engine_message = "No engine alert."
    anomaly_message = "No anomaly detected."
    
    if predictions is not None:
        max_historical = predictions["predicted_carbon_deposit"].max()
        alert_threshold = 0.7 * max_historical
        high_risk = predictions[predictions["predicted_carbon_deposit"] >= alert_threshold]
        if not high_risk.empty:
            engine_message = "⚠️ Immediate maintenance required! High carbon deposit detected."
    
    if anomalies is not None and not anomalies.empty:
        anomaly_message = "🚨 Anomaly detected! Check engine health."
    
    db_realtime.child("maintenance_alerts").update({
        "engine_alert": engine_message,
        "anomaly_alert": anomaly_message
    })
    print("\n✅ Alerts updated in Realtime Database")


# 🔄 Realtime Database Listener for Ignition

def ignition_listener(event):
    """ Callback function triggered when ignition status changes """
    print("\n🔥 Ignition status changed:", event.data)

    if event.data:  # If ignition is ON
        print("\n📊 Fetching latest sensor data...")
        df = fetch_latest_data()
        
        if df is not None:
            print("\n🔮 Running LSTM Prediction...")
            forecast_df = predict_future_values(df)
            
            if forecast_df is not None:
                df["Carbon Deposit Level"] = forecast_df["predicted_carbon_deposit"]
                print("\n🚨 Running Anomaly Detection...")
                anomalies = detect_anomalies(df)
                print("\n🔔 Checking for Maintenance Alerts...")
                trigger_maintenance_alert(forecast_df, anomalies)
    else:
        print("\n🛑 Ignition OFF - No processing needed.")

# Attach listener to Firebase Realtime Database
ignition_ref = db_realtime.child("vehicle_status/ignition")
ignition_ref.listen(ignition_listener)

print("\n🔍 Listening for ignition status updates in Realtime Database...\n")

# Keep script running
while True:
    time.sleep(10)