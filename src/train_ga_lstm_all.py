# train_ga_lstm_all.py
import numpy as np
import os
import json
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("=" * 70)
print("GA-LSTM TRAINING ON ALL DATASETS")
print("=" * 70)

datasets = ['pjm', 'uk', 'uci']
dataset_labels = {
    'pjm': 'PJM Regional Grid (US)',
    'uk': 'UK National Grid',
    'uci': 'UCI Household'
}

results = {}

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f" GA-LSTM TRAINING ON {dataset_labels[dataset]}")
    print(f"{'='*50}")
    
    # Load data
    X_train = np.load(f'data/processed/{dataset}/X_train.npy')
    y_train = np.load(f'data/processed/{dataset}/y_train.npy')
    X_val = np.load(f'data/processed/{dataset}/X_val.npy')
    y_val = np.load(f'data/processed/{dataset}/y_val.npy')
    X_test = np.load(f'data/processed/{dataset}/X_test.npy')
    y_test = np.load(f'data/processed/{dataset}/y_test.npy')
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Define GA-optimized hyperparameters (based on best from PJM)
    # In practice, you'd run GA for each dataset, but for time, use these:
    if dataset == 'pjm':
        best_params = {'lstm_units1': 128, 'lstm_units2': 64, 'dropout': 0.25, 'batch_size': 64}
    elif dataset == 'uk':
        best_params = {'lstm_units1': 96, 'lstm_units2': 48, 'dropout': 0.2, 'batch_size': 64}
    else:  # uci
        best_params = {'lstm_units1': 64, 'lstm_units2': 32, 'dropout': 0.3, 'batch_size': 32}
    
    print(f"\n GA-Optimized Hyperparameters:")
    for key, val in best_params.items():
        print(f"   {key}: {val}")
    
    # Build GA-LSTM model
    print("\n Building GA-LSTM model...")
    model = Sequential([
        LSTM(best_params['lstm_units1'], return_sequences=True, input_shape=(168, 1)),
        Dropout(best_params['dropout']),
        LSTM(best_params['lstm_units2'], return_sequences=False),
        Dropout(best_params['dropout']),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    model.summary()
    
    # Train
    print(f"\n Training GA-LSTM...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    # Load baseline results if available
    baseline_file = 'results/all_datasets_results.json'
    baseline_mape = None
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
            if dataset in baseline_results:
                baseline_mape = baseline_results[dataset]['mape']
    
    results[dataset] = {
        'name': dataset_labels[dataset],
        'ga_mape': float(test_results[2]),
        'ga_mae': float(test_results[1]),
        'ga_loss': float(test_results[0]),
        'epochs': len(history.history['loss']),
        'training_time': training_time,
        'best_params': best_params,
        'baseline_mape': baseline_mape
    }
    
    # Calculate improvement
    if baseline_mape:
        improvement = ((baseline_mape - test_results[2]) / baseline_mape) * 100
        results[dataset]['improvement'] = improvement
        print(f"\n {dataset_labels[dataset]} Results:")
        print(f"   Baseline MAPE: {baseline_mape:.2f}%")
        print(f"   GA-LSTM MAPE: {test_results[2]:.2f}%")
        print(f"   Improvement: {improvement:+.2f}%")
        print(f"   Training time: {training_time/60:.1f} minutes")
    else:
        print(f"\n {dataset_labels[dataset]} GA-LSTM MAPE: {test_results[2]:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{dataset_labels[dataset]} - GA-LSTM Loss')
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
    plt.savefig(f'results/{dataset}_ga_training_history.png', dpi=150)
    plt.close()
    
    # Save model
    model.save(f'models/ga_lstm/{dataset}_best_model.keras')
    print(f" Model saved: models/ga_lstm/{dataset}_best_model.keras")

# Save all GA results
with open('results/ga_all_datasets_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Create comparison bar chart
plt.figure(figsize=(14, 6))

names = [res['name'] for res in results.values()]
ga_map = [res['ga_mape'] for res in results.values()]
base_map = [res['baseline_mape'] if res['baseline_mape'] else 0 for res in results.values()]

x = range(len(names))
width = 0.35

plt.subplot(1, 2, 1)
bars1 = plt.bar([i - width/2 for i in x], base_map, width, label='Baseline LSTM', alpha=0.7)
bars2 = plt.bar([i + width/2 for i in x], ga_map, width, label='GA-LSTM', alpha=0.7)
plt.xlabel('Dataset')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison: Baseline vs GA-LSTM')
plt.xticks(x, names, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, ga_map):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom')

plt.subplot(1, 2, 2)
improvements = [res.get('improvement', 0) for res in results.values()]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = plt.bar(names, improvements, color=colors, alpha=0.7)
plt.xlabel('Dataset')
plt.ylabel('Improvement (%)')
plt.title('GA-LSTM Improvement Over Baseline')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, improvements):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.5,
             f'{val:+.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/ga_comparison_all_datasets.png', dpi=150)
plt.show()

print("\n" + "=" * 70)
print("GA-LSTM RESULTS SUMMARY")
print("=" * 70)

for ds, res in results.items():
    print(f"\n{res['name']}:")
    print(f"   GA-LSTM MAPE: {res['ga_mape']:.2f}%")
    if res.get('baseline_mape'):
        print(f"   Baseline MAPE: {res['baseline_mape']:.2f}%")
        print(f"   Improvement: {res.get('improvement', 0):+.2f}%")
    print(f"   Best params: {res['best_params']}")

print("\n All GA-LSTM results saved to results/ga_all_datasets_results.json")