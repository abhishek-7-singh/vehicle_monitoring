import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('generated_data.csv')

# Display basic dataset info
print("Dataset Info:\n", df.info())
print("\nDataset Summary Statistics:\n", df.describe())

# --- DISTRIBUTION OF FEATURES ---
sns.set_theme(style="whitegrid")

# Pairplot for feature relationships
sns.pairplot(df, diag_kind="kde", plot_kws={'alpha':0.6, 's':10})
plt.show()

# --- FEATURE CORRELATION HEATMAP ---
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# --- BOX PLOTS FOR OUTLIER DETECTION ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot for Outlier Detection')
plt.show()

# --- INTERACTIVE ALTAR CHART (CO vs NOx with Carbon Deposit Level) ---
chart = alt.Chart(df).mark_circle(size=60).encode(
    x='CO (mg/km)',
    y='NOx (mg/km)',
    color='Carbon Deposit Level:N',
    tooltip=['CO (mg/km)', 'NOx (mg/km)', 'Carbon Deposit Level']
).interactive()

chart.show()

# --- TIME SERIES TREND (Assuming we have a 'Timestamp' column) ---
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    time_chart = alt.Chart(df).mark_line().encode(
        x='Timestamp:T',
        y='CO (mg/km)',
        tooltip=['Timestamp', 'CO (mg/km)']
    ).interactive()
    time_chart.show()

# --- STANDARDIZATION FOR ML ---
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_scaled['Carbon Deposit Level'] = df['Carbon Deposit Level']

# Save preprocessed data
df_scaled.to_csv('preprocessed_data.csv', index=False)
print("✅ Preprocessed dataset saved as 'preprocessed_data.csv'")