import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

print("=" * 60)
print("TRAINING LSTM ON ALL THREE DATASETS")
print("=" * 60)

# Create results folder
os.makedirs('results', exist_ok=True)

datasets = ['pjm', 'uk', 'uci']
dataset_labels = {
    'pjm': 'PJM Regional Grid (US)',
    'uk': 'UK National Grid',
    'uci': 'UCI Household'
}

results = {}

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f" TRAINING ON {dataset_labels[dataset]}")
    print(f"{'='*50}")
    
    # Check if data exists
    if not os.path.exists(f'data/processed/{dataset}/X_train.npy'):
        print(f" Data for {dataset} not found. Skipping...")
        continue
    
    # Load data
    X_train = np.load(f'data/processed/{dataset}/X_train.npy')
    y_train = np.load(f'data/processed/{dataset}/y_train.npy')
    X_val = np.load(f'data/processed/{dataset}/X_val.npy')
    y_val = np.load(f'data/processed/{dataset}/y_val.npy')
    X_test = np.load(f'data/processed/{dataset}/X_test.npy')
    y_test = np.load(f'data/processed/{dataset}/y_test.npy')
    
    print(f"\n Data shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Build model
    print("\n Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(168, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    # Train
    print(f"\n Training on {dataset_labels[dataset]}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("\n Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    results[dataset] = {
        'name': dataset_labels[dataset],
        'mape': float(test_results[2]),
        'mae': float(test_results[1]),
        'loss': float(test_results[0]),
        'epochs': len(history.history['loss']),
        'training_time': training_time
    }
    
    print(f"\n {dataset_labels[dataset]} Results:")
    print(f"   MAPE: {test_results[2]:.2f}%")
    print(f"   MAE: {test_results[1]:.4f}")
    print(f"   Training time: {training_time/60:.1f} minutes")
    
    # Plot training history
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{dataset_labels[dataset]} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Val')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mape'], label='Train')
    plt.plot(history.history['val_mape'], label='Val')
    plt.title('MAPE (%)')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_training_history.png', dpi=150)
    plt.close()
    print(f" Training plot saved: results/{dataset}_training_history.png")

# Save all results
with open('results/all_datasets_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Print final comparison
print("\n" + "=" * 60)
print("FINAL RESULTS - ALL DATASETS")
print("=" * 60)
print(f"\n{'Dataset':<25} {'MAPE (%)':<12} {'MAE':<12} {'Time (min)':<12}")
print("-" * 60)

for ds, res in results.items():
    print(f"{res['name']:<25} {res['mape']:>10.2f}% {res['mae']:>10.4f} {res['training_time']/60:>10.1f}")

# Create comparison bar chart
plt.figure(figsize=(12, 5))

names = [res['name'] for res in results.values()]
mapes = [res['mape'] for res in results.values()]

plt.subplot(1, 2, 1)
bars = plt.bar(names, mapes, color=['blue', 'green', 'orange'], alpha=0.7)
plt.title('MAPE Comparison Across Datasets')
plt.ylabel('MAPE (%)')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mapes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom')

plt.subplot(1, 2, 2)
times = [res['training_time']/60 for res in results.values()]
bars = plt.bar(names, times, color=['blue', 'green', 'orange'], alpha=0.7)
plt.title('Training Time Comparison')
plt.ylabel('Time (minutes)')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}m', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/datasets_comparison.png', dpi=150)
plt.show()

print("\n All results saved to results/ folder")
print("=" * 60)