import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import pickle

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data/processed/uk', exist_ok=True)

print("=" * 60)
print("PROCESSING UK NATIONAL GRID DATA")
print("=" * 60)

# Path to UK data
uk_folder = 'data/raw/uk_grid/'
combined_file = 'data/raw/uk_grid_combined.csv'

# Check if combined file exists, if not combine all CSV files
if not os.path.exists(combined_file):
    print("\n Combining UK data files...")
    csv_files = glob.glob(os.path.join(uk_folder, '*.csv'))
    
    if not csv_files:
        print(f" No CSV files found in {uk_folder}")
        print("Please place your UK Grid data files in this folder")
        exit()
    
    df_list = []
    for file in csv_files:
        print(f"   Reading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        df_list.append(df)
    
    uk_df = pd.concat(df_list, ignore_index=True)
    print(f" Combined {len(csv_files)} files, total rows: {len(uk_df):,}")
    
    # Save combined file
    uk_df.to_csv(combined_file, index=False)
    print(f" Saved combined file to: {combined_file}")
else:
    print(f"\n Loading combined UK data...")
    uk_df = pd.read_csv(combined_file)
    print(f" Loaded {len(uk_df):,} records")

# Display column names
print(f"\n Columns in UK dataset:")
for col in uk_df.columns:
    print(f"   - {col}")

# Identify load column (usually 'ND' for National Demand)
load_col = None
for col in uk_df.columns:
    if 'ND' in col or 'DEMAND' in col.upper() or 'LOAD' in col.upper():
        load_col = col
        break

if load_col is None:
    print("\n Could not identify load column. Please check column names above.")
    print("   Using first numeric column as load...")
    for col in uk_df.columns:
        if uk_df[col].dtype in ['float64', 'int64']:
            load_col = col
            break

print(f"\n Using load column: {load_col}")

# Create datetime column
print("\n Creating datetime column...")

if 'SETTLEMENT_DATE' in uk_df.columns and 'SETTLEMENT_PERIOD' in uk_df.columns:
    # Convert to datetime (half-hourly data)
    uk_df['datetime'] = pd.to_datetime(uk_df['SETTLEMENT_DATE']) + \
                        pd.to_timedelta((uk_df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
    print(f"   Created datetime from SETTLEMENT_DATE and SETTLEMENT_PERIOD")
elif 'datetime' in uk_df.columns:
    uk_df['datetime'] = pd.to_datetime(uk_df['datetime'])
    print(f"   Using existing datetime column")
else:
    print(" Could not create datetime. Using index as time...")
    uk_df['datetime'] = pd.date_range(start='2018-01-01', periods=len(uk_df), freq='30min')

# Sort by datetime
uk_df = uk_df.sort_values('datetime').reset_index(drop=True)

print(f"   Date range: {uk_df['datetime'].min()} to {uk_df['datetime'].max()}")

# Select and clean target column
uk_df['load'] = pd.to_numeric(uk_df[load_col], errors='coerce')
uk_df = uk_df.dropna(subset=['load'])
print(f" Cleaned load data: {len(uk_df):,} records")

# Resample to hourly (take mean of half-hourly readings)
print("\n Resampling to hourly...")
uk_hourly = uk_df.set_index('datetime')['load'].resample('1H').mean().reset_index()
uk_hourly = uk_hourly.dropna()
print(f"   Hourly records: {len(uk_hourly):,}")

# Quick statistics
print(f"\n Load statistics (MW):")
print(uk_hourly['load'].describe())

# Create visualizations
print("\n Creating visualizations...")
plt.figure(figsize=(15, 8))

# Time series (sampled)
plt.subplot(2, 2, 1)
plt.plot(uk_hourly['datetime'].iloc[::100], uk_hourly['load'].iloc[::100], linewidth=0.5, color='blue')
plt.title('UK National Grid Demand (Sampled)')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
plt.grid(True, alpha=0.3)

# Distribution
plt.subplot(2, 2, 2)
plt.hist(uk_hourly['load'], bins=100, edgecolor='black', alpha=0.7, color='green')
plt.title('Demand Distribution')
plt.xlabel('Demand (MW)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Hourly pattern
plt.subplot(2, 2, 3)
hourly_avg = uk_hourly.groupby(uk_hourly['datetime'].dt.hour)['load'].mean()
plt.bar(hourly_avg.index, hourly_avg.values, color='orange')
plt.title('Average Hourly Demand')
plt.xlabel('Hour of Day')
plt.ylabel('Average Demand (MW)')
plt.grid(True, alpha=0.3)

# Monthly pattern
plt.subplot(2, 2, 4)
monthly_avg = uk_hourly.groupby(uk_hourly['datetime'].dt.month)['load'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(months, monthly_avg.values, color='red')
plt.title('Average Monthly Demand')
plt.xlabel('Month')
plt.ylabel('Average Demand (MW)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/uk_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Plots saved to: plots/uk_exploration.png")

# Create LSTM sequences
print("\n Creating LSTM sequences...")

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(uk_hourly['load'].values.reshape(-1, 1))

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
np.save('data/processed/uk/X_train.npy', X_train)
np.save('data/processed/uk/y_train.npy', y_train)
np.save('data/processed/uk/X_val.npy', X_val)
np.save('data/processed/uk/y_val.npy', y_val)
np.save('data/processed/uk/X_test.npy', X_test)
np.save('data/processed/uk/y_test.npy', y_test)

# Save scaler
with open('data/processed/uk/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n UK data saved to data/processed/uk/")

# Summary
print("\n" + "=" * 60)
print("UK DATASET SUMMARY")
print("=" * 60)
print(f"Original records: {len(uk_df):,} (half-hourly)")
print(f"Hourly records: {len(uk_hourly):,}")
print(f"Training sequences: {len(X_train):,}")
print(f"Validation sequences: {len(X_val):,}")
print(f"Testing sequences: {len(X_test):,}")
print(f"Mean demand: {uk_hourly['load'].mean():.0f} MW")
print(f"Peak demand: {uk_hourly['load'].max():.0f} MW")
print("=" * 60)