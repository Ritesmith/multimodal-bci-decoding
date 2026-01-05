#!/usr/bin/env python3

"""
Final test to verify all fixes are working correctly
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Test all imports are working without errors"""
    print("Testing safe imports...")
    
    try:
        from src.models import MultiModalFusionModel, EEGCNNLSTM, TORCH_AVAILABLE
        print(f"✓ Models module: TORCH_AVAILABLE = {TORCH_AVAILABLE}")
    except Exception as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from src.training import BCITrainer, CrossValidationTrainer, MODULE_AVAILABLE
        print(f"✓ Training module: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False
    
    try:
        from src.evaluation import BCI_Evaluator, MODULE_AVAILABLE
        print(f"✓ Evaluation module: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    try:
        from src.realtime import RealTimeBCI, BCIDataSimulator, MODULE_AVAILABLE
        print(f"✓ Realtime module: MODULE_AVAILABLE = {MODULE_AVAILABLE}")
    except Exception as e:
        print(f"✗ Realtime import failed: {e}")
        return False
    
    try:
        from src.experiment import ExperimentRunner
        print("✓ Experiment module imported successfully")
    except Exception as e:
        print(f"✗ Experiment import failed: {e}")
        return False
    
    try:
        import main
        print("✓ main.py imported successfully")
    except Exception as e:
        print(f"✗ main.py import failed: {e}")
        return False
    
    return True

def test_graceful_degradation():
    """Test that modules gracefully handle missing dependencies"""
    print("\nTesting graceful degradation...")
    
    try:
        # Try to create a model when torch is not available
        from src.models import SafeMultiModalFusionModel
        model = SafeMultiModalFusionModel()
        print("✗ SafeMultiModalFusionModel should raise ImportError when torch unavailable")
        return False
    except ImportError:
        print("✓ SafeMultiModalFusionModel correctly raises ImportError")
    except Exception as e:
        print(f"✗ SafeMultiModalFusionModel unexpected error: {e}")
        return False
    
    try:
        # Try to create trainer when dependencies are not available
        from src.training import SafeBCITrainer
        trainer = SafeBCITrainer(None)
        print("✗ SafeBCITrainer should raise ImportError when dependencies unavailable")
        return False
    except ImportError:
        print("✓ SafeBCITrainer correctly raises ImportError")
    except Exception as e:
        print(f"✗ SafeBCITrainer unexpected error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("BCI PROJECT - FINAL VERIFICATION TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test graceful degradation
    degradation_ok = test_graceful_degradation()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"✓ Imports Working: {imports_ok}")
    print(f"✓ Graceful Degradation: {degradation_ok}")
    
    if imports_ok and degradation_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("The BCI project now has:")
        print("  • Safe imports that don't crash")
        print("  • Graceful degradation with helpful error messages")
        print("  • Wrapper classes that check dependencies")
        print("  • Proper error handling throughout")
        print("\nThe project can now be run even with missing dependencies!")
        return True
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)