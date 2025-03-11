import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define BS VI emission standards (mg/km)
standards = {
    'CO': 80,
    'NOx': 80,
    'HC_NOx': 170,
    'PM': 4.5
}

# Define realistic exhaust temperature ranges (°C)
exhaust_temp_ranges = {
    'Normal': (200, 400),
    'Medium': (400, 600),
    'High': (600, 800)
}

# Define ranges for each carbon deposit level
ranges = {
    'Normal': {
        'CO': (0, standards['CO']),
        'NOx': (0, standards['NOx']),
        'HC_NOx': (0, standards['HC_NOx']),
        'PM': (0, standards['PM']),
        'Vibration': (0.5, 1.5),
        'ExhaustTemp': exhaust_temp_ranges['Normal']
    },
    'Medium': {
        'CO': (standards['CO'], 1.5 * standards['CO']),
        'NOx': (standards['NOx'], 1.5 * standards['NOx']),
        'HC_NOx': (standards['HC_NOx'], 1.5 * standards['HC_NOx']),
        'PM': (standards['PM'], 1.5 * standards['PM']),
        'Vibration': (1.5, 2.5),
        'ExhaustTemp': exhaust_temp_ranges['Medium']
    },
    'High': {
        'CO': (1.5 * standards['CO'], 2 * standards['CO']),
        'NOx': (1.5 * standards['NOx'], 2 * standards['NOx']),
        'HC_NOx': (1.5 * standards['HC_NOx'], 2 * standards['HC_NOx']),
        'PM': (1.5 * standards['PM'], 2 * standards['PM']),
        'Vibration': (2.5, 3.5),
        'ExhaustTemp': exhaust_temp_ranges['High']
    }
}

# Function to generate data with noise and outliers
def generate_data(n, low, high, noise_factor=0.2, outlier_fraction=0.02):
    data = np.random.uniform(low, high, n)
    noise = np.random.normal(0, noise_factor * (high - low), n)
    data += noise
    num_outliers = int(outlier_fraction * n)
    outliers = np.random.uniform(low - 0.5 * (high - low), high + 0.5 * (high - low), num_outliers)
    data[:num_outliers] = outliers
    return np.clip(data, low, high)

# Generate dataset
n_samples = 100000
df = pd.DataFrame()

for level, params in ranges.items():
    num_samples = n_samples if level != 'High' else int(n_samples * 0.5)
    temp_df = pd.DataFrame({
        'CO (mg/km)': generate_data(num_samples, *params['CO']),
        'NOx (mg/km)': generate_data(num_samples, *params['NOx']),
        'HC + NOx (mg/km)': generate_data(num_samples, *params['HC_NOx']),
        'PM (mg/km)': generate_data(num_samples, *params['PM']),
        'Vibration (mm/s²)': generate_data(num_samples, *params['Vibration']),
        'Exhaust Temperature (°C)': generate_data(num_samples, *params['ExhaustTemp']),
        'Carbon Deposit Level': level
    })
    df = pd.concat([df, temp_df], ignore_index=True)

# Encode labels
df['Carbon Deposit Level'] = LabelEncoder().fit_transform(df['Carbon Deposit Level'])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset to CSV
df.to_csv('generated_data.csv', index=False)

print("Data generation complete. Saved as 'generated_data.csv'.")