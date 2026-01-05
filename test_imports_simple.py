#!/usr/bin/env python3

# Test imports to check if fixes work
print("Testing imports...")

try:
    from src.models import MultiModalFusionModel, EEGCNNLSTM, fNIRSFCN, TORCH_AVAILABLE
    print(f"✓ Models import successful, TORCH_AVAILABLE: {TORCH_AVAILABLE}")
except Exception as e:
    print(f"✗ Models import failed: {e}")

try:
    from src.training import BCITrainer, CrossValidationTrainer, MODULE_AVAILABLE
    print(f"✓ Training import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Training import failed: {e}")

try:
    from src.evaluation import BCI_Evaluator, MODULE_AVAILABLE
    print(f"✓ Evaluation import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Evaluation import failed: {e}")

try:
    from src.realtime import RealTimeBCI, BCIDataSimulator, MODULE_AVAILABLE
    print(f"✓ Realtime import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Realtime import failed: {e}")

print("\nTesting main import...")
try:
    from src.experiment import ExperimentRunner
    print("✓ Experiment import successful")
except Exception as e:
    print(f"✗ Experiment import failed: {e}")

print("Import test complete!")