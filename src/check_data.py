import os
import glob

raw_folder = 'data/raw/'

if not os.path.exists(raw_folder):
    print(f" Folder not found. Creating...")
    os.makedirs(raw_folder, exist_ok=True)

csv_files = glob.glob(os.path.join(raw_folder, '*.csv'))

print(f"\n Raw data folder: {os.path.abspath(raw_folder)}")
print(f"\n Found {len(csv_files)} CSV files:")

for i, file in enumerate(csv_files):
    size = os.path.getsize(file) / (1024*1024)
    print(f"{i+1}. {os.path.basename(file)} - {size:.2f} MB")

if not csv_files:
    print("\n  No CSV files found!")
    print("Please place your PJM files in:", os.path.abspath(raw_folder))