import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

print("=" * 60)
print("PROCESSING UCI DATA FOR LSTM")
print("=" * 60)

# Create directory
os.makedirs('data/processed/uci', exist_ok=True)

# Load UCI data
uci_file = 'data/raw/uci_household/household_power_consumption.txt'

if not os.path.exists(uci_file):
    print(" UCI file not found")
    exit()

print(" Loading UCI data (this may take a minute)...")

# Load in chunks for memory efficiency
chunk_size = 100000
chunks = []

for chunk in pd.read_csv(uci_file, sep=';', na_values=['?'], chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df = df.dropna()
print(f" Loaded {len(df):,} records")

# Create datetime
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.sort_values('datetime')
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Use Global_active_power
target = 'Global_active_power'
print(f" Target: {target}")
print(f"   Range: {df[target].min():.3f} - {df[target].max():.3f} kW")

# Resample to hourly
print("\n Resampling to hourly...")
df_hourly = df.set_index('datetime')[target].resample('1H').mean().reset_index()
df_hourly = df_hourly.dropna()
print(f" Hourly records: {len(df_hourly):,}")

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_hourly[target].values.reshape(-1, 1))

# Create sequences
sequence_length = 168
X, y = [], []

print(f"\n Creating sequences (length={sequence_length})...")
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length])

X = np.array(X)
y = np.array(y)
print(f" Sequences: {X.shape}")

# Split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n Data split:")
print(f"   Training: {len(X_train):,}")
print(f"   Validation: {len(X_val):,}")
print(f"   Testing: {len(X_test):,}")

# Save
np.save('data/processed/uci/X_train.npy', X_train)
np.save('data/processed/uci/y_train.npy', y_train)
np.save('data/processed/uci/X_val.npy', X_val)
np.save('data/processed/uci/y_val.npy', y_val)
np.save('data/processed/uci/X_test.npy', X_test)
np.save('data/processed/uci/y_test.npy', y_test)

with open('data/processed/uci/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n UCI data saved to data/processed/uci/")