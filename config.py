"""
Configuration file for Multi-modal BCI Decoding Project
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# EEG configuration
EEG_CONFIG = {
    "sampling_rate": 1000,  # Hz
    "notch_freq": 50,       # Hz (power line noise)
    "bandpass_low": 0.5,    # Hz
    "bandpass_high": 40,    # Hz
    "channels": 64,         # number of EEG channels
    "epoch_tmin": -2.0,     # seconds before event
    "epoch_tmax": 5.0,      # seconds after event
}

# fNIRS configuration
FNIRS_CONFIG = {
    "sampling_rate": 10,     # Hz
    "bandpass_low": 0.01,    # Hz
    "bandpass_high": 0.5,    # Hz
    "channels": 20,          # number of fNIRS channels
    "roi_count": 10,         # number of regions of interest
    "hemodynamic_delay": 5.0, # seconds
}

# Data preprocessing parameters
PREPROCESSING = {
    "ica_n_components": 20,
    "artifact_threshold": 3.0,  # standard deviations
    "baseline_correction": True,
    "time_alignment": True,
}

# Feature extraction parameters
FEATURES = {
    "eeg": {
        "wavelet_freqs": [4, 8, 13, 30],  # frequency bands
        "time_windows": [0.5, 1.0, 2.0],   # seconds
        "plv_threshold": 0.3,
    },
    "fnirs": {
        "window_size": 2.0,  # seconds
        "overlap": 0.5,       # overlap ratio
        "smoothing": True,
    }
}

# Model architecture parameters
MODEL_CONFIG = {
    "eeg_stream": {
        "input_channels": EEG_CONFIG["channels"],
        "input_length": 1000,  # time points
        "conv_filters": [32, 64, 128],
        "conv_kernel": (3, 3),
        "lstm_units": [128, 64],
        "dropout": 0.3,
    },
    "fnirs_stream": {
        "input_channels": FNIRS_CONFIG["roi_count"],
        "input_length": 200,   # time points (after downsampling)
        "dense_units": [128, 64],
        "dropout": 0.2,
    },
    "fusion": {
        "attention_units": 64,
        "output_classes": 4,    # 4-class motor imagery
        "fusion_method": "attention",  # "attention", "concat", "weighted"
    }
}

# Training parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
}

# Cross-validation and evaluation
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
    "stratify": True,
    "metrics": ["accuracy", "precision", "recall", "f1", "auc", "itr"],
    "sliding_window_size": 200,  # ms for real-time prediction
}

# Real-time constraints
REALTIME_CONFIG = {
    "max_inference_time": 200,  # ms
    "update_frequency": 5,       # Hz
    "buffer_size": 1000,         # samples
}

# Device configuration
DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "bci_project.log",
}