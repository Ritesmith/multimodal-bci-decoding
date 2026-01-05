import numpy as np
from pathlib import Path
import scipy.io as sio

data_dir = Path('data/raw')
subject_dir = data_dir / 'subject_01'

# Create complete sample data with all expected fields
eeg_data = np.random.randn(10, 4, 100) * 1e-6
fnirs_data = np.random.rand(10, 2, 50) * 0.1
events = np.array([[i*50, 0, i%4] for i in range(10)])

# Create complete EEG data file
eeg_complete = {
    'eeg_data': eeg_data,
    'fs': 250,  # sample rate
    'channel_names': [f'EEG_{i:03d}' for i in range(4)],
    'n_trials': 10,
    'subject_id': 1
}

# Create complete fNIRS data file
fnirs_complete = {
    'fnirs_data': fnirs_data,
    'fs': 10,  # sample rate for fNIRS
    'channel_names': [f'fNIRS_{i:02d}' for i in range(2)],
    'n_trials': 10,
    'subject_id': 1
}

# Create complete events file
events_complete = {
    'events': events,
    'class_labels': events[:, 2],
    'n_trials': 10,
    'subject_id': 1
}

sio.savemat(subject_dir / 'eeg.mat', eeg_complete)
sio.savemat(subject_dir / 'fnirs.mat', fnirs_complete)
sio.savemat(subject_dir / 'events.mat', events_complete)

print('Created complete sample data for subject 01')