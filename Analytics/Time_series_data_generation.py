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

# Define carbon deposit progression over time
carbon_growth_factor = {
    'Normal': 1.002,  # Slow increase
    'Medium': 1.005,  # Moderate increase
    'High': 1.01      # Faster increase
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

# Function to simulate time-series data
def generate_time_series_data(n_samples, level, initial_values, time_step=1):
    time_stamps = np.arange(0, n_samples * time_step, time_step)  # Simulated timestamps
    data = { "Time": time_stamps }

    for feature, (low, high) in ranges[level].items():
        values = [np.random.uniform(low, high)]  # Start with a random initial value

        for _ in range(n_samples - 1):
            # Simulate gradual increase in emissions due to carbon build-up
            next_value = values[-1] * carbon_growth_factor[level]
            next_value += np.random.normal(0, 0.02 * (high - low))  # Add some noise
            values.append(min(next_value, high))  # Clip to max range

        data[feature] = values

    return pd.DataFrame(data)

# Generate dataset
n_samples = 5000  # Number of time steps per engine
df = pd.DataFrame()

for level in ranges.keys():
    temp_df = generate_time_series_data(n_samples, level, initial_values=None)
    temp_df['Carbon Deposit Level'] = level  # Add label
    df = pd.concat([df, temp_df], ignore_index=True)

# Encode labels
df['Carbon Deposit Level'] = LabelEncoder().fit_transform(df['Carbon Deposit Level'])

# Save dataset to CSV
df.to_csv('time_series_data.csv', index=False)

print("Time-series data generation complete. Saved as 'time_series_data.csv'.")