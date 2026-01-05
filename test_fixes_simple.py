import os
import sys
sys.path.insert(0, 'D:/Stazica/my_files/University_works/self study/脑机接口课程遐想')

print("Testing safe imports...")

try:
    from src.models import MultiModalFusionModel, EEGCNNLSTM, TORCH_AVAILABLE
    print(f"✓ Models import successful, TORCH_AVAILABLE: {TORCH_AVAILABLE}")
    if not TORCH_AVAILABLE:
        print("  (expected behavior - torch not installed)")
except Exception as e:
    print(f"✗ Models import failed: {e}")

try:
    from src.training import BCITrainer, CrossValidationTrainer, MODULE_AVAILABLE
    print(f"✓ Training import successful, MODULE_AVAILABLE: {MODULE_AVAILABLE}")
    if not MODULE_AVAILABLE:
        print("  (expected behavior - some dependencies missing)")
except Exception as e:
    print(f"✗ Training import failed: {e}")

print("Test complete!")