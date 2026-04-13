import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)

print("=" * 60)
print("PREPARING DATA FOR LSTM MODEL")
print("=" * 60)

# Read in chunks to avoid memory issues
chunk_size = 50000
chunks = []
print("\n Reading data in chunks...")

for i, chunk in enumerate(pd.read_csv('data/raw/pjm_combined_2018_2025.csv', 
                                      chunksize=chunk_size,
                                      usecols=['datetime_beginning_ept', 'mw'])):
    print(f"   Processing chunk {i+1}...")
    chunk['datetime'] = pd.to_datetime(chunk['datetime_beginning_ept'])
    chunks.append(chunk[['datetime', 'mw']])

# Combine all chunks
df = pd.concat(chunks, ignore_index=True)
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\n Total records: {len(df):,}")

# Create sequences for LSTM
def create_sequences(data, seq_length=168):  # 168 hours = 1 week
    """Create input sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['mw'].values.reshape(-1, 1))

# Save scaler for later use
with open('models/scalers/minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(" Scaler saved")

# Create sequences
print("\n Creating LSTM sequences...")
sequence_length = 168  # 1 week
X, y = create_sequences(scaled_data.flatten(), sequence_length)

print(f" Input shape: {X.shape}")
print(f" Target shape: {y.shape}")

# Split into train/val/test (chronologically)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\n Data split:")
print(f"   Training: {X_train.shape[0]:,} sequences")
print(f"   Validation: {X_val.shape[0]:,} sequences")
print(f"   Testing: {X_test.shape[0]:,} sequences")

# Save processed data
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)

print("\n All data saved to 'data/processed/'")


#.....................................................................

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create directories
os.makedirs('models/baseline', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("=" * 60)
print("BASELINE LSTM MODEL FOR ENERGY FORECASTING")
print("=" * 60)

# Load preprocessed data
print("\n Loading preprocessed data...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"\n Data shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_val: {X_val.shape}")
print(f"   y_val: {y_val.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")

# Reshape data for LSTM (if needed)
# X_train shape should be (samples, timesteps, features)
if len(X_train.shape) == 2:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\n Reshaped for LSTM:")
print(f"   X_train: {X_train.shape}")

# Build baseline LSTM model
print("\n Building baseline LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(168, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

print(model.summary())

# Callbacks - FIXED: using .keras extension instead of .h5
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('models/baseline/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train model
print("\n Training baseline LSTM...")
print("   This may take 30-60 minutes depending on your hardware...")

start_time = datetime.now()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

training_time = datetime.now() - start_time
print(f"\n Training completed in {training_time}")

# Plot training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

# MAE plot
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

# MAPE plot
plt.subplot(1, 3, 3)
plt.plot(history.history['mape'], label='Training MAPE')
plt.plot(history.history['val_mape'], label='Validation MAPE')
plt.title('Mean Absolute Percentage Error')
plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_training_history.png', dpi=150)
plt.show()

# Evaluate on test set
print("\n Evaluating on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"\n   Test Loss (MSE): {test_results[0]:.4f}")
print(f"   Test MAE: {test_results[1]:.4f}")
print(f"   Test MAPE: {test_results[2]:.4f}%")

# Save results
with open('results/baseline_results.txt', 'w') as f:
    f.write("BASELINE LSTM RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Training completed: {datetime.now()}\n")
    f.write(f"Training time: {training_time}\n\n")
    f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}\n")
    f.write(f"Final validation MAPE: {history.history['val_mape'][-1]:.4f}%\n\n")
    f.write(f"Test loss (MSE): {test_results[0]:.4f}\n")
    f.write(f"Test MAE: {test_results[1]:.4f}\n")
    f.write(f"Test MAPE: {test_results[2]:.4f}%\n")

print("\n Results saved to results/baseline_results.txt")

# Make predictions on test set
print("\n Making predictions on test set...")
y_pred = model.predict(X_test)

# Plot predictions vs actual
plt.figure(figsize=(15, 6))

# Plot first 500 predictions
plt.plot(y_test[:500], label='Actual', linewidth=1.5, alpha=0.7)
plt.plot(y_pred[:500], label='Predicted', linewidth=1.5, alpha=0.7)
plt.title('LSTM Predictions vs Actual (First 500 Test Points)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Load')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/baseline_predictions.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print(" BASELINE MODEL COMPLETE!")
print("=" * 60)
print("\n Output files:")
print("   - models/baseline/best_model.keras (trained model)")
print("   - results/baseline_training_history.png (training plots)")
print("   - results/baseline_predictions.png (prediction vs actual)")
print("   - results/baseline_results.txt (numerical results)")