import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

print("=" * 60)
print("PROCESSING UK DATA FOR LSTM")
print("=" * 60)

# Create directory
os.makedirs('data/processed/uk', exist_ok=True)

# Load combined UK data
uk_file = 'data/raw/uk_grid_combined.csv'

if not os.path.exists(uk_file):
    print(" UK combined file not found")
    exit()

print(" Loading UK data...")
df = pd.read_csv(uk_file)
print(f" Loaded {len(df):,} records")

# Display columns
print(f"\n Columns: {list(df.columns)}")

# Identify columns
datetime_col = None
load_col = None

for col in df.columns:
    if 'settlement_date' in col.lower() or 'date' in col.lower():
        datetime_col = col
    if 'nd' in col.lower() or 'demand' in col.lower():
        load_col = col

print(f"\n Using:")
print(f"   Datetime column: {datetime_col}")
print(f"   Load column: {load_col}")

# Create datetime
if 'SETTLEMENT_DATE' in df.columns and 'SETTLEMENT_PERIOD' in df.columns:
    df['datetime'] = pd.to_datetime(df['SETTLEMENT_DATE']) + \
                     pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
    print(" Created datetime from SETTLEMENT_DATE + SETTLEMENT_PERIOD")

# Sort and clean
df = df.sort_values('datetime')
df['load'] = pd.to_numeric(df[load_col], errors='coerce')
df = df.dropna(subset=['load'])
print(f" Cleaned: {len(df):,} records")

# Resample to hourly
df_hourly = df.set_index('datetime')['load'].resample('1H').mean().reset_index()
df_hourly = df_hourly.dropna()
print(f" Hourly records: {len(df_hourly):,}")
print(f"   Date range: {df_hourly['datetime'].min()} to {df_hourly['datetime'].max()}")
print(f"   Load range: {df_hourly['load'].min():.0f} - {df_hourly['load'].max():.0f} MW")

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_hourly['load'].values.reshape(-1, 1))

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
np.save('data/processed/uk/X_train.npy', X_train)
np.save('data/processed/uk/y_train.npy', y_train)
np.save('data/processed/uk/X_val.npy', X_val)
np.save('data/processed/uk/y_val.npy', y_val)
np.save('data/processed/uk/X_test.npy', X_test)
np.save('data/processed/uk/y_test.npy', y_test)

with open('data/processed/uk/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n UK data saved to data/processed/uk/")