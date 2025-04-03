import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import joblib  # For saving/loading scalers

# Load dataset
df = pd.read_csv('/kaggle/working/time_series_data.csv')

# Define feature and target columns
features = ['CO', 'NOx', 'HC_NOx', 'PM', 'Vibration', 'ExhaustTemp']
target = ['Carbon Deposit Level']

#  **Separate Feature and Target Scalers**
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

#  **Fit and Transform**
df[features] = scaler_features.fit_transform(df[features])
df[target] = scaler_target.fit_transform(df[target])

# 🔹 **Save the scalers for later use**
joblib.dump(scaler_features, "/kaggle/working/scaler_features.pkl")
joblib.dump(scaler_target, "/kaggle/working/scaler_target.pkl")

# Define sequence length
SEQ_LENGTH = 10

# Convert dataset into sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :-1]  # Use all features except the last column (target)
        label = data[i+seq_length, -1]   # Predict the next Carbon Deposit Level
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Prepare the data
data = df[features + target].values
X, y = create_sequences(data, SEQ_LENGTH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define DataLoader for batching
train_data = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define the LSTM Model with Batch Normalization
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
        out = self.batch_norm(out[:, -1, :])  # Normalize last timestep output
        out = self.fc(out)
        return out

# Model Parameters
input_dim = len(features)  # Number of input features
hidden_dim = 128  # Increased hidden units
num_layers = 2
output_dim = 1

# Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CarbonLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0005)  # RMSprop works better for LSTMs

# Training Loop
num_epochs = 100  # Increased from 50
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient Clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

# 🔹 **Save Model**
torch.save(model.state_dict(), "/kaggle/working/carbon_lstm_model.pth")
print(" Model training completed and saved!")

# 🔹 **Load Model & Evaluate**
model.eval()
X_test_torch = X_test_torch.to(device)
y_test_torch = y_test_torch.to(device)

with torch.no_grad():
    y_pred = model(X_test_torch)

# Convert predictions back to NumPy
y_pred = y_pred.cpu().numpy().flatten()
y_test = y_test_torch.cpu().numpy().flatten()

# 🔹 **Load the Correct Scaler for Inverse Transform**
scaler_target = joblib.load('/kaggle/working/scaler_target.pkl')
y_pred = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Rounding predictions for classification metrics
y_pred_rounded = np.round(y_pred)

# Print sample predictions
print("Actual:", y_test[:10])
print("Predicted:", y_pred[:10])
print("Rounded Predicted:", y_pred_rounded[:10])

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
accuracy = accuracy_score(y_test, y_pred_rounded)
precision = precision_score(y_test, y_pred_rounded, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_rounded, average='weighted', zero_division=1)

print(f"\n📊 **Evaluation Metrics:**")
print(f" Mean Squared Error (MSE): {mse:.4f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")