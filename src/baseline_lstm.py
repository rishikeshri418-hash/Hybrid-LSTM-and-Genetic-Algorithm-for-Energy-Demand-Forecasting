# check_uk_data.py
import os
import glob

print("=" * 60)
print("CHECKING UK DATA")
print("=" * 60)

# Look for UK data
uk_paths = [
    'data/raw/uk_grid_combined.csv',
    'data/raw/uk_grid/*.csv',
    'data/raw/uk/*.csv',
    'data/raw/uk_grid/historic_demand*.csv'
]

found = False
for pattern in uk_paths:
    files = glob.glob(pattern)
    if files:
        print(f"\n✅ Found UK data at: {pattern}")
        for f in files[:3]:  # Show first 3
            size = os.path.getsize(f) / (1024 * 1024)
            print(f"   - {os.path.basename(f)} ({size:.2f} MB)")
        found = True

if not found:
    print("\n❌ No UK data found. Please download from:")
    print("   https://www.neso.energy/data-portal")
    print("   Search for 'Historic Demand Data'")