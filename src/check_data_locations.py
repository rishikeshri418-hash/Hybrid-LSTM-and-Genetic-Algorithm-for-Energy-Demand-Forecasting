# check_raw_data.py
import os
import glob

print("=" * 60)
print("CHECKING RAW DATA FOLDERS")
print("=" * 60)

# Check raw folder
raw_path = 'data/raw/'
if os.path.exists(raw_path):
    print(f"\n📁 {raw_path}")
    for item in os.listdir(raw_path):
        item_path = os.path.join(raw_path, item)
        if os.path.isdir(item_path):
            print(f"   📂 {item}/")
            files = os.listdir(item_path)
            for f in files[:5]:  # Show first 5
                size = os.path.getsize(os.path.join(item_path, f)) / (1024 * 1024)
                print(f"      📄 {f} ({size:.2f} MB)")
        else:
            size = os.path.getsize(item_path) / (1024 * 1024)
            print(f"   📄 {item} ({size:.2f} MB)")
else:
    print(f"❌ {raw_path} not found")

# Also check for UK data specifically
print("\n" + "=" * 60)
print("SEARCHING FOR UK DATA")
print("=" * 60)

uk_patterns = [
    'data/raw/*uk*',
    'data/raw/*UK*',
    'data/raw/*demand*',
    'data/raw/*grid*',
    'data/raw/uk_grid*',
]

for pattern in uk_patterns:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        print(f"\n✅ Found: {pattern}")
        for m in matches[:3]:
            if os.path.isfile(m):
                size = os.path.getsize(m) / (1024 * 1024)
                print(f"   - {os.path.basename(m)} ({size:.2f} MB)")
            else:
                print(f"   - {m}/")

print("\n" + "=" * 60)
print("SEARCHING FOR UCI DATA")
print("=" * 60)

uci_patterns = [
    'data/raw/*uci*',
    'data/raw/*household*',
    'data/raw/*power*',
    'data/raw/uci_household*',
]

for pattern in uci_patterns:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        print(f"\n✅ Found: {pattern}")
        for m in matches[:3]:
            if os.path.isfile(m):
                size = os.path.getsize(m) / (1024 * 1024)
                print(f"   - {os.path.basename(m)} ({size:.2f} MB)")
            else:
                print(f"   - {m}/")