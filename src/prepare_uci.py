import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

# Create directories FIRST
os.makedirs('plots', exist_ok=True)
os.makedirs('data/processed/uci', exist_ok=True)

print("=" * 60)
print("PROCESSING UCI HOUSEHOLD POWER CONSUMPTION DATA")
print("=" * 60)

# Load UCI data
uci_file = 'data/raw/uci_household/household_power_consumption.txt'
print(f"\n Loading UCI dataset...")

# Check if file exists
if not os.path.exists(uci_file):
    print(f" File not found: {uci_file}")
    exit()

# Load in chunks to handle large file
chunk_size = 100000
chunks = []

for i, chunk in enumerate(pd.read_csv(uci_file, sep=';', na_values=['?'], chunksize=chunk_size)):
    print(f"   Loading chunk {i+1}...")
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f" Loaded {len(df):,} records")

# Handle missing values
print("\n Handling missing values...")
missing_before = df.isnull().sum().sum()
df = df.dropna()
missing_after = df.isnull().sum().sum()
print(f"   Removed {missing_before - missing_after:,} rows with missing values")
print(f"   Remaining: {len(df):,} records")

# Create datetime column
print("\n Creating datetime column...")
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Sort by datetime
df = df.sort_values('datetime').reset_index(drop=True)

# Check for duplicates
duplicates = df['datetime'].duplicated().sum()
if duplicates > 0:
    print(f"   Found {duplicates} duplicate timestamps, removing...")
    df = df.drop_duplicates(subset=['datetime'])

# Use Global_active_power as target (kW)
target_col = 'Global_active_power'
print(f"\n Target variable: {target_col}")
print(f"   Range: {df[target_col].min():.3f} - {df[target_col].max():.3f} kW")
print(f"   Mean: {df[target_col].mean():.3f} kW")

# Resample to hourly (take mean) for consistency with other datasets
print("\n Resampling from minute to hourly...")
df_hourly = df.set_index('datetime')[target_col].resample('1H').mean().reset_index()
df_hourly = df_hourly.dropna()
print(f"   Hourly records: {len(df_hourly):,}")
print(f"   Date range: {df_hourly['datetime'].min()} to {df_hourly['datetime'].max()}")

# Quick visualization
print("\n Creating visualizations...")
plt.figure(figsize=(15, 8))

# Daily pattern (first week of 2008)
week_data = df_hourly[(df_hourly['datetime'] >= '2008-01-01') & 
                       (df_hourly['datetime'] < '2008-01-08')]

plt.subplot(2, 2, 1)
plt.plot(week_data['datetime'], week_data[target_col], 'b-', linewidth=1)
plt.title('UCI Household Power - First Week of 2008')
plt.xlabel('Date')
plt.ylabel('Power (kW)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Distribution
plt.subplot(2, 2, 2)
plt.hist(df_hourly[target_col], bins=100, edgecolor='black', alpha=0.7)
plt.title('Power Consumption Distribution')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Hourly pattern
plt.subplot(2, 2, 3)
hourly_avg = df_hourly.groupby(df_hourly['datetime'].dt.hour)[target_col].mean()
plt.bar(hourly_avg.index, hourly_avg.values)
plt.title('Average Hourly Consumption')
plt.xlabel('Hour of Day')
plt.ylabel('Average Power (kW)')
plt.grid(True, alpha=0.3)

# Monthly pattern
plt.subplot(2, 2, 4)
monthly_avg = df_hourly.groupby(df_hourly['datetime'].dt.month)[target_col].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(months, monthly_avg.values)
plt.title('Average Monthly Consumption')
plt.xlabel('Month')
plt.ylabel('Average Power (kW)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/uci_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Plots saved to: plots/uci_exploration.png")

# Create LSTM sequences
print("\n Creating LSTM sequences...")

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_hourly[target_col].values.reshape(-1, 1))

# Create sequences (168 hours = 1 week)
sequence_length = 168
X, y = [], []

for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length])

X = np.array(X)
y = np.array(y)

print(f" Input shape: {X.shape}")
print(f" Target shape: {y.shape}")

# Split chronologically
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n Data split:")
print(f"   Training: {len(X_train):,} sequences")
print(f"   Validation: {len(X_val):,} sequences")
print(f"   Testing: {len(X_test):,} sequences")

# Save processed data
np.save('data/processed/uci/X_train.npy', X_train)
np.save('data/processed/uci/y_train.npy', y_train)
np.save('data/processed/uci/X_val.npy', X_val)
np.save('data/processed/uci/y_val.npy', y_val)
np.save('data/processed/uci/X_test.npy', X_test)
np.save('data/processed/uci/y_test.npy', y_test)

# Save scaler
with open('data/processed/uci/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n UCI data saved to data/processed/uci/")

# Summary statistics
print("\n" + "=" * 60)
print("UCI DATASET SUMMARY")
print("=" * 60)
print(f"Original records: {len(df):,} (minute-level)")
print(f"Hourly records: {len(df_hourly):,}")
print(f"Training sequences: {len(X_train):,}")
print(f"Validation sequences: {len(X_val):,}")
print(f"Testing sequences: {len(X_test):,}")
print(f"Mean consumption: {df_hourly[target_col].mean():.3f} kW")
print(f"Peak consumption: {df_hourly[target_col].max():.3f} kW")
print("=" * 60)