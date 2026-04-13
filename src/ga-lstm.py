# check_results.py
import numpy as np
import json
import os

print("=" * 50)
print("CHECKING YOUR MODEL RESULTS")
print("=" * 50)

# Check if comparison file exists
if os.path.exists('results/quick_comparison.json'):
    with open('results/quick_comparison.json', 'r') as f:
        results = json.load(f)
    
    print("\n RESULTS FOUND:")
    print(json.dumps(results, indent=2))
    
    if 'improvement_percent' in results:
        print(f"\n GA-LSTM improved baseline by {results['improvement_percent']:.2f}%")
else:
    print("\n No results file found yet")

# Check if we can load the GA model and get its performance directly
try:
    from tensorflow.keras.models import load_model
    import numpy as np
    
    print("\n Loading models to get fresh metrics...")
    
    # Load test data (subsample for speed)
    X_test = np.load('data/processed/X_test.npy')[:5000]
    y_test = np.load('data/processed/y_test.npy')[:5000]
    
    # Load baseline
    baseline = load_model('models/baseline/best_model.keras')
    baseline_pred = baseline.predict(X_test, verbose=0)
    
    # Calculate baseline MAPE
    baseline_mape = np.mean(np.abs((y_test - baseline_pred.flatten()) / (y_test + 1e-8))) * 100
    print(f"\n Baseline MAPE: {baseline_mape:.2f}%")
    
    # Check if GA model exists
    ga_path = 'models/ga_lstm/best_model.keras'
    if os.path.exists(ga_path):
        ga_model = load_model(ga_path)
        ga_pred = ga_model.predict(X_test, verbose=0)
        ga_mape = np.mean(np.abs((y_test - ga_pred.flatten()) / (y_test + 1e-8))) * 100
        print(f" GA-LSTM MAPE: {ga_mape:.2f}%")
        
        improvement = ((baseline_mape - ga_mape) / baseline_mape) * 100
        print(f"\n Improvement: {improvement:.2f}%")
    else:
        print("\n GA model not found at:", ga_path)
        
except Exception as e:
    print(f"\n Error loading models: {e}")

print("\n" + "=" * 50)