import sys
print(f" Python version: {sys.version}")

print("\n Testing packages...")
try:
    import pandas as pd
    print(f" pandas {pd.__version__}")
except: print(" pandas failed")

try:
    import numpy as np
    print(f" numpy {np.__version__}")
except: print(" numpy failed")

try:
    import tensorflow as tf
    print(f" tensorflow {tf.__version__}")
except: print(" tensorflow failed")

try:
    from deap import base
    print(f" deap installed")
except: print(" deap failed")

print("\n Setup complete!")