#!/usr/bin/env python3
"""
Test script to check if the module import fixes work
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("Testing module imports...")

try:
    print("Testing data_loader...")
    from data_loader import check_module_availability
    available = check_module_availability()
    print(f"Data loader module available: {available}")
except ImportError as e:
    print(f"data_loader import failed: {e}")

try:
    print("Testing feature_extraction...")
    from feature_extraction import check_module_availability
    available = check_module_availability()
    print(f"Feature extraction module available: {available}")
except ImportError as e:
    print(f"feature_extraction import failed: {e}")

try:
    print("Testing models...")
    from models import check_module_availability
    available = check_module_availability()
    print(f"Models module available: {available}")
except ImportError as e:
    print(f"models import failed: {e}")

try:
    print("Testing training...")
    from training import check_module_availability
    available = check_module_availability()
    print(f"Training module available: {available}")
except ImportError as e:
    print(f"training import failed: {e}")

try:
    print("Testing evaluation...")
    from evaluation import check_module_availability
    available = check_module_availability()
    print(f"Evaluation module available: {available}")
except ImportError as e:
    print(f"evaluation import failed: {e}")

try:
    print("Testing realtime...")
    from realtime import check_module_availability
    available = check_module_availability()
    print(f"Realtime module available: {available}")
except ImportError as e:
    print(f"realtime import failed: {e}")

try:
    print("Testing experiment...")
    from experiment import ExperimentRunner, ExperimentConfig
    print("Experiment module imported successfully")
except ImportError as e:
    print(f"experiment import failed: {e}")

print("\nTest completed!")