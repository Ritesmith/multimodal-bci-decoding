#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing safe imports...")

try:
    from src.models import MultiModalFusionModel, EEGCNNLSTM, TORCH_AVAILABLE
    print(f"✓ Models import successful, TORCH_AVAILABLE: {TORCH_AVAILABLE}")
    if not TORCH_AVAILABLE:
        print("  (expected behavior - torch not installed)")
except Exception as e:
    print(f"✗ Models import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.training import BCITrainer, CrossValidationTrainer, MODULE_AVAILABLE
    print(f"✓ Training import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
    if not MODULE_AVAILABLE:
        print("  (expected behavior - some dependencies missing)")
except Exception as e:
    print(f"✗ Training import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.evaluation import BCI_Evaluator, MODULE_AVAILABLE
    print(f"✓ Evaluation import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
    if not MODULE_AVAILABLE:
        print("  (expected behavior - some dependencies missing)")
except Exception as e:
    print(f"✗ Evaluation import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.realtime import RealTimeBCI, BCIDataSimulator, MODULE_AVAILABLE
    print(f"✓ Realtime import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
    if not MODULE_AVAILABLE:
        print("  (expected behavior - torch not installed)")
except Exception as e:
    print(f"✗ Realtime import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting main import...")
try:
    from src.experiment import ExperimentRunner
    print("✓ Experiment import successful")
except Exception as e:
    print(f"✗ Experiment import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing main.py...")
try:
    import main
    print("✓ main.py import successful")
except Exception as e:
    print(f"✗ main.py import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test complete!")