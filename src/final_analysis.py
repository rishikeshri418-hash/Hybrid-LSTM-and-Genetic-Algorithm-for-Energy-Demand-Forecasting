# final_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import json

# Your results
datasets = ['PJM Regional Grid (US)', 'UK National Grid', 'UCI Household']
baseline = [1.57, 4.46, 58.81]
ga_lstm = [1.49, 5.82, 57.46]
improvements = [5.03, -30.61, 2.29]
training_times = [72.6, 21.1, 37.7]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: MAPE Comparison
x = np.arange(len(datasets))
width = 0.35
axes[0].bar(x - width/2, baseline, width, label='Baseline LSTM', alpha=0.7, color='blue')
axes[0].bar(x + width/2, ga_lstm, width, label='GA-LSTM', alpha=0.7, color='red')
axes[0].set_ylabel('MAPE (%)')
axes[0].set_title('MAPE Comparison: Baseline vs GA-LSTM')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets, rotation=15)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (b, g) in enumerate(zip(baseline, ga_lstm)):
    axes[0].text(i - width/2, b + 1, f'{b:.2f}%', ha='center', va='bottom', fontsize=9)
    axes[0].text(i + width/2, g + 1, f'{g:.2f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement Percentage
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = axes[1].bar(datasets, improvements, color=colors, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_ylabel('Improvement (%)')
axes[1].set_title('GA-LSTM Improvement Over Baseline')
axes[1].set_xticklabels(datasets, rotation=15)
axes[1].grid(True, alpha=0.3, axis='y')

for bar, imp in zip(bars, improvements):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if imp > 0 else -3),
                 f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)

# Plot 3: Training Time
bars = axes[2].bar(datasets, training_times, color='purple', alpha=0.7)
axes[2].set_ylabel('Training Time (minutes)')
axes[2].set_title('Training Time Comparison')
axes[2].set_xticklabels(datasets, rotation=15)
axes[2].grid(True, alpha=0.3, axis='y')

for bar, t in zip(bars, training_times):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{t:.1f} min', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/final_comparison_all.png', dpi=150)
plt.show()

# Print discussion points
print("\n" + "=" * 70)
print("DISCUSSION POINTS FOR YOUR FINAL PROJECT REPORT")
print("=" * 70)

print("\n1. KEY FINDINGS:")
print("    GA-LSTM improved PJM by 5.03% (1.57% → 1.49%)")
print("    GA-LSTM underperformed on UK (-30.61%)")
print("    GA-LSTM slightly improved UCI by 2.29% (58.81% → 57.46%)")

print("\n2. WHY UK UNDERPERFORMED (For Discussion):")
print("   • Different grid characteristics (UK vs US)")
print("   • GA may have overfitted to validation set")
print("   • UK data resampled from half-hourly to hourly")
print("   • Shorter training time (21 min vs 72 min for PJM)")

print("\n3. CONTRIBUTIONS TO KNOWLEDGE:")
print("   • Demonstrated GA optimization works best for stable, patterned data")
print("   • Showed importance of dataset-specific hyperparameter tuning")
print("   • Highlighted limitations of GA for highly variable data")

print("\n4. RECOMMENDATIONS FOR FUTURE WORK:")
print("   • Use different architectures for different dataset types")
print("   • Implement adaptive GA with dataset-specific search spaces")
print("   • Combine GA with other optimization methods")

# Save results to JSON
results = {
    'datasets': datasets,
    'baseline_mape': baseline,
    'ga_mape': ga_lstm,
    'improvement': improvements,
    'training_times': training_times
}

with open('results/final_results_discussion.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n Results saved to results/final_results_discussion.json")