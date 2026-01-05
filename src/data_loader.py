"""
Data loading and management for multi-modal BCI project
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

# Safe imports for optional dependencies
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    mne = None

try:
    from scipy.io import loadmat
    import h5py
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    loadmat = None
    h5py = None

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EEG_CONFIG, FNIRS_CONFIG

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = MNE_AVAILABLE and SCIPY_AVAILABLE


class BNCIDataLoader:
    """Loader for BNCI Horizon 2020 Dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.eeg_config = EEG_CONFIG
        self.fnirs_config = FNIRS_CONFIG
        
    def load_subject_data(self, subject_id: int) -> Dict:
        """Load data for a specific subject"""
        subject_path = self.data_path / f"subject_{subject_id:02d}"
        
        # Load EEG data
        eeg_file = subject_path / "eeg.mat"
        fnirs_file = subject_path / "fnirs.mat"
        events_file = subject_path / "events.mat"
        
        if not all([f.exists() for f in [eeg_file, fnirs_file, events_file]]):
            raise FileNotFoundError(f"Data files not found for subject {subject_id}")
        
        # Load EEG
        eeg_data = self._load_eeg(eeg_file)
        
        # Load fNIRS
        fnirs_data = self._load_fnirs(fnirs_file)
        
        # Load events
        events = self._load_events(events_file)
        
        return {
            'eeg': eeg_data,
            'fnirs': fnirs_data,
            'events': events,
            'subject_id': subject_id
        }
    
    def _load_eeg(self, file_path: Path) -> Dict:
        """Load EEG data from .mat file"""
        try:
            mat_data = loadmat(file_path)
            eeg_raw = mat_data['eeg_data']
            fs = mat_data['fs'][0, 0]
            
            # Create MNE Raw object
            ch_names = [f'EEG {i:03d}' for i in range(eeg_raw.shape[0])]
            ch_types = ['eeg'] * eeg_raw.shape[0]
            
            info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
            raw = mne.io.RawArray(eeg_raw, info, verbose=False)
            
            return {
                'raw': raw,
                'data': eeg_raw,
                'sampling_rate': fs,
                'channels': ch_names
            }
        except Exception as e:
            logger.error(f"Error loading EEG data: {e}")
            raise
    
    def _load_fnirs(self, file_path: Path) -> Dict:
        """Load fNIRS data from .mat file"""
        try:
            mat_data = loadmat(file_path)
            fnirs_raw = mat_data['fnirs_data']
            fs = mat_data['fs'][0, 0]
            
            # Separate oxy and deoxy hemoglobin
            if fnirs_raw.shape[0] % 2 == 0:
                half_channels = fnirs_raw.shape[0] // 2
                hbo = fnirs_raw[:half_channels, :]
                hbr = fnirs_raw[half_channels:, :]
            else:
                # If not clearly separated, use all data
                hbo = fnirs_raw
                hbr = np.zeros_like(fnirs_raw)
            
            return {
                'hbo': hbo,
                'hbr': hbr,
                'data': fnirs_raw,
                'sampling_rate': fs,
                'channels': len(hbo)
            }
        except Exception as e:
            logger.error(f"Error loading fNIRS data: {e}")
            raise
    
    def _load_events(self, file_path: Path) -> Dict:
        """Load event markers"""
        try:
            mat_data = loadmat(file_path)
            events = mat_data['events']
            
            # Handle missing event_types gracefully
            if 'event_types' in mat_data:
                event_types = mat_data['event_types'].flatten()
            else:
                # Default event types for motor imagery
                event_types = np.array(['left_hand', 'right_hand', 'feet', 'tongue'])
                logger.warning(f"event_types not found, using default: {event_types}")
            
            return {
                'timestamps': events[:, 0].astype(int),  # Convert to integer timestamps
                'event_ids': events[:, 2].astype(int),  # event_id is in column 2
                'event_types': event_types,
                'events': events
            }
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            raise


class OpenBMIDataLoader:
    """Loader for OpenBMI Dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_session_data(self, session_id: str) -> Dict:
        """Load data for a specific session"""
        session_path = self.data_path / session_id
        
        # Implementation for OpenBMI dataset
        # Similar structure to BNCI loader
        pass


class DataPreprocessor:
    """Data preprocessing pipeline"""
    
    def __init__(self):
        self.eeg_config = EEG_CONFIG
        self.fnirs_config = FNIRS_CONFIG
        
    def preprocess_eeg(self, eeg_data: Dict, events: Dict) -> Dict:
        """Preprocess EEG data"""
        raw = eeg_data['raw']
        
        # Apply notch filter for power line noise
        raw.notch_filter(freqs=self.eeg_config['notch_freq'], verbose=False)
        
        # Apply bandpass filter
        raw.filter(
            l_freq=self.eeg_config['bandpass_low'],
            h_freq=self.eeg_config['bandpass_high'],
            method='fir',
            verbose=False
        )
        
        # Create epochs around events
        events_array = np.column_stack([
            events['timestamps'],
            np.zeros(len(events['timestamps']), dtype=int),
            events['event_ids']
        ]).astype(int)
        
        epochs = mne.Epochs(
            raw,
            events_array,
            tmin=self.eeg_config['epoch_tmin'],
            tmax=self.eeg_config['epoch_tmax'],
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Apply ICA for artifact removal
        try:
            ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800)
            ica.fit(epochs, verbose=False)

            # Detect and remove EOG/EMG components (if EOG channels exist)
            try:
                eog_indices, _ = ica.find_bads_eog(epochs, verbose=False)
                if eog_indices:
                    ica.exclude.extend(eog_indices)
            except ValueError:
                # No EOG channels found, skip EOG detection
                pass

            epochs_clean = ica.apply(epochs, verbose=False)
        except Exception as e:
            # If ICA fails, return original epochs
            logger.warning(f"ICA artifact removal failed: {e}, using original data")
            epochs_clean = epochs
        
        return {
            'epochs': epochs_clean,
            'data': epochs_clean.get_data(),
            'ica': ica,
            'events': events_array
        }
    
    def preprocess_fnirs(self, fnirs_data: Dict, events: Dict) -> Dict:
        """Preprocess fNIRS data"""
        hbo = fnirs_data['hbo']
        hbr = fnirs_data['hbr']
        fs = fnirs_data['sampling_rate']
        
        # Convert EEG timestamps to fNIRS timestamps
        # Assume EEG and fNIRS started at the same time
        eeg_fs = 1000.0  # EEG sampling rate
        fnirs_timestamps = (events['timestamps'] / eeg_fs * fs).astype(int)
        
        # Apply bandpass filter for hemodynamic response
        from scipy.signal import butter, filtfilt
        
        nyquist = fs / 2
        low = self.fnirs_config['bandpass_low'] / nyquist
        high = self.fnirs_config['bandpass_high'] / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        # Check if axis=1 is valid (need at least 2D array)
        if hbo.ndim == 1:
            hbo = hbo.reshape(1, -1)
            hbr = hbr.reshape(1, -1)
        
        hbo_filtered = filtfilt(b, a, hbo, axis=1)
        hbr_filtered = filtfilt(b, a, hbr, axis=1)
        
        # Baseline correction - simpler approach
        # Use the first 10% of data as baseline
        baseline_samples = max(1, int(hbo_filtered.shape[1] * 0.1))
        baseline_mean_hbo = np.mean(hbo_filtered[:, :baseline_samples], axis=1, keepdims=True)
        baseline_mean_hbr = np.mean(hbr_filtered[:, :baseline_samples], axis=1, keepdims=True)
        
        hbo_corrected = hbo_filtered - baseline_mean_hbo
        hbr_corrected = hbr_filtered - baseline_mean_hbr
        
        # Epoch extraction
        epoch_length = int((self.fnirs_config['hemodynamic_delay'] + 5.0) * fs)
        start_sample = int(-2.0 * fs)  # Start 2 seconds before event
        
        epochs_hbo = []
        epochs_hbr = []
        event_labels = []
        
        for i, timestamp in enumerate(fnirs_timestamps):
            start_idx = timestamp + start_sample
            end_idx = start_idx + epoch_length
            
            # Ensure indices are within bounds
            if start_idx >= 0 and end_idx <= hbo_corrected.shape[1]:
                epochs_hbo.append(hbo_corrected[:, start_idx:end_idx])
                epochs_hbr.append(hbr_corrected[:, start_idx:end_idx])
                event_labels.append(events['event_ids'][i])
        
        # Check if we have any valid epochs
        if len(epochs_hbo) == 0:
            # If no epochs extracted, create dummy epochs for compatibility
            logger.warning(f"No valid fNIRS epochs extracted, creating dummy data")
            epochs_hbo = [np.zeros((hbo_corrected.shape[0], epoch_length))]
            epochs_hbr = [np.zeros((hbr_corrected.shape[0], epoch_length))]
            event_labels = [1]
        
        return {
            'hbo_epochs': np.array(epochs_hbo),
            'hbr_epochs': np.array(epochs_hbr),
            'event_labels': np.array(event_labels),
            'data': {
                'hbo': hbo_corrected,
                'hbr': hbr_corrected
            }
        }
    
    def synchronize_modalities(self, eeg_epochs: Dict, fnirs_epochs: Dict) -> Dict:
        """Synchronize EEG and fNIRS epochs"""
        # Ensure both modalities have same number of trials
        min_trials = min(
            len(eeg_epochs['epochs']),
            len(fnirs_epochs['hbo_epochs'])
        )
        
        # Align by truncating to minimum
        eeg_data_aligned = eeg_epochs['data'][:min_trials]
        hbo_aligned = fnirs_epochs['hbo_epochs'][:min_trials]
        hbr_aligned = fnirs_epochs['hbr_epochs'][:min_trials]
        
        # Align labels
        eeg_labels = eeg_epochs['events'][:min_trials, 2]
        fnirs_labels = fnirs_epochs['event_labels'][:min_trials]
        
        # Verify label consistency
        if not np.array_equal(eeg_labels, fnirs_labels):
            logger.warning("Label mismatch between EEG and fNIRS data")
        
        return {
            'eeg': eeg_data_aligned,
            'hbo': hbo_aligned,
            'hbr': hbr_aligned,
            'labels': eeg_labels,
            'n_trials': min_trials
        }


def create_dataset_loader(dataset_type: str, data_path: str) -> object:
    """Factory function for data loaders"""
    if dataset_type.lower() == 'bnci':
        return BNCIDataLoader(data_path)
    elif dataset_type.lower() == 'openbmi':
        return OpenBMIDataLoader(data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def check_module_availability():
    """Check if all required dependencies are available"""
    return MODULE_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeDataPreprocessor:
    def __init__(self):
        if not check_module_availability():
            raise ImportError("Required dependencies (mne, scipy) not available for data loading module")
        self.preprocessor = DataPreprocessor()
    
    def __getattr__(self, name):
        return getattr(self.preprocessor, name)