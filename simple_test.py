import sys
print("Python path:", sys.executable)

try:
    import numpy as np
    print("numpy: OK")
except ImportError as e:
    print(f"numpy: FAILED - {e}")

try:
    import torch
    print("torch: OK")
except ImportError as e:
    print(f"torch: FAILED - {e}")

try:
    import mne
    print("mne: OK")
except ImportError as e:
    print(f"mne: FAILED - {e}")

try:
    import sklearn
    print("sklearn: OK")
except ImportError as e:
    print(f"sklearn: FAILED - {e}")

try:
    import matplotlib
    print("matplotlib: OK")
except ImportError as e:
    print(f"matplotlib: FAILED - {e}")

try:
    import pywt
    print("pywt: OK")
except ImportError as e:
    print(f"pywt: FAILED - {e}")

print("\nTest completed!")