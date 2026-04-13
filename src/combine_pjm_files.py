import pandas as pd
import glob
import os

print("=" * 50)
print("COMBINING PJM HOURLY LOAD FILES")
print("=" * 50)

# Path to your data folder
raw_folder = 'data/raw/'
output_file = 'data/raw/pjm_combined_2018_2025.csv'

# Find all CSV files
all_files = sorted(glob.glob(os.path.join(raw_folder, '*.csv')))

if not all_files:
    print(" No CSV files found!")
    exit()

print(f" Found {len(all_files)} files to combine")

# Combine all files
df_list = []
total_rows = 0

for i, file in enumerate(all_files):
    print(f"\r Reading file {i+1}/{len(all_files)}: {os.path.basename(file)}", end="")
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Add filename as source (optional)
        df['source_file'] = os.path.basename(file)
        
        rows = len(df)
        total_rows += rows
        df_list.append(df)
        
        print(f" → {rows:,} rows")
    except Exception as e:
        print(f"\n Error reading {file}: {e}")

print(f"\n Successfully read {len(df_list)} files")

if not df_list:
    print(" No files could be read!")
    exit()

# Concatenate all dataframes
print("\n Combining all files...")
combined_df = pd.concat(df_list, ignore_index=True)

print(f"\n COMBINED DATASET SUMMARY")
print("-" * 30)
print(f"Total rows: {len(combined_df):,}")
print(f"Total columns: {len(combined_df.columns)}")
print(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Show column names
print(f"\n Columns:")
for col in combined_df.columns:
    print(f"   - {col}")

# Date range if datetime columns exist
datetime_cols = [col for col in combined_df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
if datetime_cols:
    for dt_col in datetime_cols[:2]:  # Check first 2 datetime columns
        try:
            combined_df[dt_col] = pd.to_datetime(combined_df[dt_col])
            print(f"\n Date range for {dt_col}:")
            print(f"   From: {combined_df[dt_col].min()}")
            print(f"   To: {combined_df[dt_col].max()}")
            break
        except:
            pass

# Check for AEP zone data
if 'zone' in combined_df.columns:
    zone_counts = combined_df['zone'].value_counts()
    print(f"\n Zone distribution:")
    for zone, count in zone_counts.head().items():
        pct = (count/len(combined_df))*100
        print(f"   {zone}: {count:,} ({pct:.1f}%)")

# Check for missing values
missing = combined_df.isnull().sum()
if missing.sum() > 0:
    print(f"\n  Missing values found:")
    for col in missing[missing > 0].index:
        print(f"   {col}: {missing[col]:,} missing ({missing[col]/len(combined_df)*100:.1f}%)")
else:
    print(f"\n No missing values found!")

# Save combined file
print(f"\n Saving combined file to: {output_file}")
combined_df.to_csv(output_file, index=False)
print(f" Done! File size: {os.path.getsize(output_file)/(1024*1024):.2f} MB")

# Create a smaller sample for quick testing
sample_file = 'data/raw/pjm_sample_50k.csv'
combined_df.sample(n=min(50000, len(combined_df))).to_csv(sample_file, index=False)
print(f" Sample file saved: {sample_file} (50,000 random rows)")

print("\n" + "=" * 50)
print(" COMBINING COMPLETE!")
print("=" * 50)