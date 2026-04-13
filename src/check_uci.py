import pandas as pd
import os

uci_file = 'data/raw/uci_household/household_power_consumption.txt'

if os.path.exists(uci_file):
    print(" UCI dataset found!")
    
    # Load first few rows (semicolon separated, missing values as '?')
    df_sample = pd.read_csv(uci_file, sep=';', nrows=5, na_values=['?'])
    
    print(f"\n Shape preview: {df_sample.shape}")
    print("\n First 5 rows:")
    print(df_sample.to_string())
    
    print("\n Columns:")
    for col in df_sample.columns:
        print(f"   - {col}")
        
    # Check data types
    print("\n Data types:")
    print(df_sample.dtypes)
    
    # Memory estimate
    file_size = os.path.getsize(uci_file) / (1024 * 1024)
    print(f"\n File size: {file_size:.2f} MB")
    
    # Estimate total rows (fast way - count lines in file)
    with open(uci_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # subtract header
    print(f" Estimated total rows: {total_lines:,}")
    
else:
    print(" UCI dataset not found!")
    print(f"Expected location: {os.path.abspath(uci_file)}")
    print("\nPlease download from:")
    print("https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip")