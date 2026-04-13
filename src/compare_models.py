import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import json

print("=" * 60)
print("COMPARING BASELINE VS GA-OPTIMIZED LSTM")
print("=" * 60)

# Load test data
print("\n Loading test data...")
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Load both models
print("\n Loading both models...")
baseline_model = load_model('models/baseline/best_model.keras')
ga_model = load_model('models/ga_lstm/best_model.keras')

# Load GA hyperparameters if available
if os.path.exists('models/ga_lstm/best_params.json'):
    with open('models/ga_lstm/best_params.json', 'r') as f:
        best_params = json.load(f)
    print("\n🧬 Best GA Hyperparameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")

# Make predictions in batches
print("\n🔮 Generating predictions for comparison...")

def predict_in_batches(model, X, batch_size=10000):
    """Predict in batches to avoid memory issues"""
    n_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
    predictions = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X))
        batch_pred = model.predict(X[start:end], verbose=0, batch_size=1024)
        predictions.append(batch_pred)
    
    return np.concatenate(predictions, axis=0).flatten()

baseline_pred = predict_in_batches(baseline_model, X_test)
ga_pred = predict_in_batches(ga_model, X_test)

print(" Predictions generated")

# Calculate metrics for both models
print("\n" + "=" * 60)
print("MODEL COMPARISON METRICS")
print("=" * 60)

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate all metrics for a model"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE with safety
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

baseline_metrics = calculate_metrics(y_test, baseline_pred, "Baseline")
ga_metrics = calculate_metrics(y_test, ga_pred, "GA-LSTM")

# Print comparison table
print("\n Performance Comparison:")
print("-" * 60)
print(f"{'Metric':<15} {'Baseline':<15} {'GA-LSTM':<15} {'Improvement':<15}")
print("-" * 60)

for metric in ['MAE', 'RMSE', 'MAPE']:
    baseline_val = baseline_metrics[metric]
    ga_val = ga_metrics[metric]
    
    if metric == 'MAPE':
        improvement = ((baseline_val - ga_val) / baseline_val) * 100
        print(f"{metric:<15} {baseline_val:.4f}%      {ga_val:.4f}%      {improvement:+.2f}%")
    else:
        improvement = ((baseline_val - ga_val) / baseline_val) * 100
        print(f"{metric:<15} {baseline_val:.4f}      {ga_val:.4f}      {improvement:+.2f}%")

print("-" * 60)

# Save comparison results
comparison = {
    'baseline': baseline_metrics,
    'ga_lstm': ga_metrics,
    'improvement': {
        'mae_improvement': ((baseline_metrics['MAE'] - ga_metrics['MAE']) / baseline_metrics['MAE']) * 100,
        'rmse_improvement': ((baseline_metrics['RMSE'] - ga_metrics['RMSE']) / baseline_metrics['RMSE']) * 100,
        'mape_improvement': ((baseline_metrics['MAPE'] - ga_metrics['MAPE']) / baseline_metrics['MAPE']) * 100
    }
}

with open('results/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=4)

print("\n Comparison saved to: results/model_comparison.json")

# Visual comparison
print("\n Creating comparison visualizations...")

plt.figure(figsize=(16, 12))

# Plot 1: Time series comparison (subsampled)
plt.subplot(3, 2, 1)
plot_step = 20
plot_range = slice(0, 1000, plot_step)
plt.plot(y_test[plot_range], 'k-', label='Actual', linewidth=2, alpha=0.8)
plt.plot(baseline_pred[plot_range], 'b--', label='Baseline', linewidth=1.5, alpha=0.7)
plt.plot(ga_pred[plot_range], 'r-', label='GA-LSTM', linewidth=1.5, alpha=0.7)
plt.title('Predictions Comparison (Subsampled)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Load')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Baseline scatter
plt.subplot(3, 2, 2)
scatter_step = 50
scatter_range = slice(0, 5000, scatter_step)
plt.scatter(y_test[scatter_range], baseline_pred[scatter_range], alpha=0.3, s=5, c='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Baseline: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)

# Plot 3: GA-LSTM scatter
plt.subplot(3, 2, 3)
plt.scatter(y_test[scatter_range], ga_pred[scatter_range], alpha=0.3, s=5, c='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('GA-LSTM: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True, alpha=0.3)

# Plot 4: Error distributions
plt.subplot(3, 2, 4)
baseline_errors = y_test - baseline_pred
ga_errors = y_test - ga_pred
plt.hist(baseline_errors[::50], bins=50, alpha=0.5, label='Baseline', edgecolor='black')
plt.hist(ga_errors[::50], bins=50, alpha=0.5, label='GA-LSTM', edgecolor='black')
plt.title('Error Distribution Comparison')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: MAPE comparison bar chart
plt.subplot(3, 2, 5)
models = ['Baseline', 'GA-LSTM']
mape_values = [baseline_metrics['MAPE'], ga_metrics['MAPE']]
colors = ['blue', 'red']
bars = plt.bar(models, mape_values, color=colors, alpha=0.7)
plt.title('MAPE Comparison')
plt.ylabel('MAPE (%)')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, mape_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom')

# Plot 6: Improvement gauge
plt.subplot(3, 2, 6)
improvement = comparison['improvement']['mape_improvement']
plt.pie([abs(improvement), 100 - abs(improvement) if improvement < 100 else 0], 
        labels=['Improvement', 'Remaining'], 
        colors=['green', 'lightgray'],
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        startangle=90)
plt.title(f'MAPE Improvement: {improvement:.2f}%')

plt.tight_layout()
plt.savefig('results/model_comparison_plots.png', dpi=150)
plt.show()

print("\n Comparison plots saved to: results/model_comparison_plots.png")

# Summary
print("\n" + "=" * 60)
print("SUMMARY: GA-LSTM vs BASELINE")
print("=" * 60)

if ga_metrics['MAPE'] < baseline_metrics['MAPE']:
    improvement = ((baseline_metrics['MAPE'] - ga_metrics['MAPE']) / baseline_metrics['MAPE']) * 100
    print(f"\n GA-LSTM outperformed baseline by {improvement:.2f}% in MAPE!")
    print(f"   Baseline MAPE: {baseline_metrics['MAPE']:.2f}%")
    print(f"   GA-LSTM MAPE: {ga_metrics['MAPE']:.2f}%")
else:
    print(f"\n Baseline still better. Let's analyze why...")

print("\n All results saved to 'results/' folder")
print("=" * 60)