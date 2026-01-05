import sys
sys.path.insert(0, 'D:/Stazica/my_files/University_works/self study/脑机接口课程遐想')

print("Testing all module imports...")

try:
    from src.models import MultiModalFusionModel, TORCH_AVAILABLE
    print(f"✓ Models: TORCH_AVAILABLE = {TORCH_AVAILABLE}")
except Exception as e:
    print(f"✗ Models failed: {e}")

try:
    from src.training import BCITrainer, CrossValidationTrainer, MODULE_AVAILABLE
    print(f"✓ Training: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Training failed: {e}")

try:
    from src.evaluation import BCI_Evaluator, MODULE_AVAILABLE
    print(f"✓ Evaluation: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Evaluation failed: {e}")

try:
    from src.realtime import RealTimeBCI, BCIDataSimulator, MODULE_AVAILABLE
    print(f"✓ Realtime: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
except Exception as e:
    print(f"✗ Realtime failed: {e}")

try:
    from src.experiment import ExperimentRunner
    print("✓ Experiment import successful")
except Exception as e:
    print(f"✗ Experiment failed: {e}")

try:
    import main
    print("✓ main.py import successful")
except Exception as e:
    print(f"✗ main.py failed: {e}")

print("All tests completed!")