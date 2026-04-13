import numpy as np
from tensorflow.keras.models import load_model, save_model
import os
import json

print("=" * 60)
print("SAVING GA-LSTM MODEL")
print("=" * 60)

# Create directory if it doesn't exist
os.makedirs('models/ga_lstm', exist_ok=True)

# Since you just trained it, it should still be in memory
# But if you closed Python, you'll need to load from the latest checkpoint

# Option 1: If you have the model in memory from your training script
try:
    # Try to access the model from the training script's namespace
    # This may not work if you've restarted Python
    from your_training_script import model as ga_model
    print(" Found model in memory")
except:
    # Option 2: Find the latest checkpoint
    print("\n🔍 Searching for latest checkpoint...")
    
    # Check common checkpoint locations
    checkpoint_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.keras') or file.endswith('.h5'):
                checkpoint_files.append(os.path.join(root, file))
    
    if checkpoint_files:
        # Get the most recent file
        latest_file = max(checkpoint_files, key=os.path.getctime)
        print(f" Found latest model: {latest_file}")
        ga_model = load_model(latest_file)
    else:
        print(" No model files found. Please train GA-LSTM first.")
        exit()

# Save to the standard location
save_path = 'models/ga_lstm/best_model.keras'
ga_model.save(save_path)
print(f" Model saved to: {save_path}")

# Also save in older format for compatibility
ga_model.save('models/ga_lstm/best_model.h5', save_format='h5')
print(f" Model also saved as: models/ga_lstm/best_model.h5")

# Create a dummy hyperparameter file (replace with your actual GA results)
best_params = {
    'lstm_layers': 2,
    'units_per_layer': 128,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'optimizer': 'adam',
    'best_epoch': 33,
    'final_val_loss': 0.00045  # Replace with your actual value
}

with open('models/ga_lstm/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f" Hyperparameters saved to: models/ga_lstm/best_params.json")
print("\n File contents:")
print(json.dumps(best_params, indent=2))

print("\n" + "=" * 60)
print(" GA-LSTM MODEL SAVED SUCCESSFULLY!")
print("=" * 60)
print("\nNow you can run compare_models.py")