try:
    import torch
    print("torch is available")
except ImportError as e:
    print(f"torch not available: {e}")

try:
    import sklearn
    print("sklearn is available")
except ImportError as e:
    print(f"sklearn not available: {e}")

try:
    import matplotlib
    print("matplotlib is available")
except ImportError as e:
    print(f"matplotlib not available: {e}")

try:
    import mne
    print("mne is available")
except ImportError as e:
    print(f"mne not available: {e}")

print("Test complete")