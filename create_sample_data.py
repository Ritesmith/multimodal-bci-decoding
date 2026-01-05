#!/usr/bin/env python3

"""
Create sample data for testing the BCI project
"""
import numpy as np
import os
from pathlib import Path
from scipy.io import savemat

def create_sample_data():
    """Create sample BCI data files"""
    
    # Create data directory structure
    base_dir = Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data for 20 subjects
    n_subjects = 20
    n_channels_eeg = 64
    n_channels_fnirs = 16
    n_trials = 40
    n_timepoints_eeg = 1000
    n_timepoints_fnirs = 200
    sample_rate = 250
    
    print(f"Creating sample data for {n_subjects} subjects...")
    
    for subject_id in range(1, n_subjects + 1):
        print(f"Creating data for subject {subject_id}...")
        
        # Create subject directory
        subject_dir = base_dir / f"subject_{subject_id:02d}"
        subject_dir.mkdir(exist_ok=True)
        
        # Generate sample EEG data (trials × channels × timepoints)
        eeg_data = np.random.randn(n_trials, n_channels_eeg, n_timepoints_eeg) * 1e-6  # Convert to microvolts
        
        # Add some realistic EEG features
        for trial in range(n_trials):
            # Add alpha waves (8-12 Hz)
            t = np.linspace(0, n_timepoints_eeg/sample_rate, n_timepoints_eeg)
            alpha_wave = 2e-6 * np.sin(2 * np.pi * 10 * t)
            eeg_data[trial, :, :] += alpha_wave
            
            # Add beta waves (13-30 Hz)
            beta_wave = 1e-6 * np.sin(2 * np.pi * 20 * t)
            eeg_data[trial, :, :] += beta_wave
        
        # Generate sample fNIRS data (trials × channels × timepoints)
        fnirs_data = np.random.rand(n_trials, n_channels_fnirs, n_timepoints_fnirs) * 0.1
        
        # Add hemodynamic response patterns
        for trial in range(n_trials):
            # Simulate hemodynamic response for different brain activities
            t = np.linspace(0, n_timepoints_fnirs/10, n_timepoints_fnirs)  # fNIRS usually slower
            hrf = 0.05 * np.exp(-((t - 5)**2) / 2)  # Gaussian-like response
            
            # Different patterns for different channels
            for ch in range(n_channels_fnirs):
                fnirs_data[trial, ch, :] += hrf * (0.5 + 0.5 * np.sin(ch * np.pi / 8))
        
        # Generate events (trial markers and labels)
        events = np.zeros((n_trials, 3), dtype=int)
        for trial in range(n_trials):
            events[trial, 0] = trial * (n_timepoints_eeg // 2)  # Sample time
            events[trial, 1] = 0  # Event type
            events[trial, 2] = trial % 4  # Class label (0-3)
        
        # Create class labels
        class_labels = events[:, 2]
        
        # Save data files
        savemat(subject_dir / "eeg.mat", {
            'eeg_data': eeg_data,
            'fs': sample_rate,  # 修复字段名
            'channel_names': [f'EEG_{i:03d}' for i in range(n_channels_eeg)],
            'n_trials': n_trials
        })
        
        savemat(subject_dir / "fnirs.mat", {
            'fnirs_data': fnirs_data,
            'fs': 10,  # 修复字段名，fNIRS usually lower sample rate
            'channel_names': [f'fNIRS_{i:02d}' for i in range(n_channels_fnirs)],
            'n_trials': n_trials
        })
        
        savemat(subject_dir / "events.mat", {
            'timestamps': events[:, 0],  # 修复字段名
            'event_ids': events[:, 2],   # 修复字段名
            'events': events,
            'class_labels': class_labels,
            'n_trials': n_trials
        })
        
        print(f"  ✓ Created EEG data: {eeg_data.shape}")
        print(f"  ✓ Created fNIRS data: {fnirs_data.shape}")
        print(f"  ✓ Created events: {events.shape}")
    
    print(f"\n✅ Sample data creation completed!")
    print(f"   Created data for {n_subjects} subjects")
    print(f"   Data saved to: {base_dir.absolute()}")
    print(f"\nNow you can run: python main.py --mode experiment")

if __name__ == "__main__":
    create_sample_data()