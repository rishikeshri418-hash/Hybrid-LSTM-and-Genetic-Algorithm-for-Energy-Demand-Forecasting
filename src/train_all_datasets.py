import os

print("=" * 60)
print("CHECKING DATA LOCATIONS")
print("=" * 60)

# Check various possible locations
locations_to_check = [
    'data/processed/',
    'data/processed/pjm/',
    'data/processed/uk/',
    'data/processed/uci/',
    'data/raw/',
]

for loc in locations_to_check:
    if os.path.exists(loc):
        print(f"\n {loc} exists")
        files = os.listdir(loc)
        print(f"   Files: {files[:5]}")  # Show first 5 files
    else:
        print(f"\n {loc} does not exist")

# Specifically check for your processed data files
print("\n" + "=" * 60)
print("CHECKING FOR SPECIFIC FILES")
print("=" * 60)

datasets = ['pjm', 'uk', 'uci']
for dataset in datasets:
    path = f'data/processed/{dataset}/X_train.npy'
    if os.path.exists(path):
        print(f" {dataset}: X_train.npy found at {path}")
    else:
        print(f" {dataset}: X_train.npy NOT found")
        
        # Check alternative locations
        alt_path = f'data/processed/{dataset.upper()}/X_train.npy'
        if os.path.exists(alt_path):
            print(f"    Found at: {alt_path}")