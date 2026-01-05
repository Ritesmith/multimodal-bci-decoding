"""
Feature extraction for multi-modal BCI data
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Safe imports for optional dependencies
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    pywt = None

try:
    from scipy.signal import hilbert, coherence
    from scipy.stats import skew, kurtosis
    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False
    hilbert = None
    coherence = None
    skew = None
    kurtosis = None

from config import EEG_CONFIG, FNIRS_CONFIG, FEATURES

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = PYWT_AVAILABLE and SCIPY_SIGNAL_AVAILABLE


class EEGFeatureExtractor:
    """Feature extraction for EEG signals"""
    
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        self.freq_bands = FEATURES['eeg']['wavelet_freqs']
        
    def extract_time_frequency_features(self, eeg_epochs: np.ndarray) -> Dict:
        """
        Extract time-frequency features using Morlet wavelets
        
        Args:
            eeg_epochs: Shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            Dictionary containing time-frequency features
        """
        n_trials, n_channels, n_timepoints = eeg_epochs.shape
        
        # Define frequency bands
        freqs = np.array([4, 8, 13, 30])  # theta, alpha, beta, gamma boundaries
        
        features = {
            'power': np.zeros((n_trials, n_channels, len(freqs)-1)),
            'phase': np.zeros((n_trials, n_channels, len(freqs)-1)),
            'complex_features': np.zeros((n_trials, n_channels, len(freqs)-1))
        }
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = eeg_epochs[trial_idx, ch_idx, :]
                
                for band_idx in range(len(freqs)-1):
                    # Morlet wavelet transform
                    freq_range = (freqs[band_idx], freqs[band_idx+1])
                    center_freq = np.mean(freq_range)
                    
                    # Use continuous wavelet transform
                    scales = pywt.frequency2scale('cmor3-1.0', [center_freq])
                    cwt, freqs_used = pywt.cwt(signal, scales, 'cmor3-1.0', 
                                               sampling_period=1/self.sampling_rate)
                    
                    # Extract power and phase
                    wavelet_coeffs = cwt[0, :]  # First (and only) scale
                    
                    # Average power in frequency band
                    power = np.mean(np.abs(wavelet_coeffs)**2)
                    phase = np.angle(wavelet_coeffs)
                    
                    # Complex features (magnitude and phase combined)
                    complex_feat = wavelet_coeffs.real + 1j * wavelet_coeffs.imag
                    
                    features['power'][trial_idx, ch_idx, band_idx] = power
                    features['phase'][trial_idx, ch_idx, band_idx] = np.mean(phase)
                    features['complex_features'][trial_idx, ch_idx, band_idx] = np.mean(complex_feat)
        
        return features
    
    def compute_phase_locking_value(self, eeg_epochs: np.ndarray) -> np.ndarray:
        """
        Compute Phase Locking Value (PLV) between EEG channels
        
        Args:
            eeg_epochs: Shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            PLV matrix shape (n_trials, n_channels, n_channels)
        """
        n_trials, n_channels, n_timepoints = eeg_epochs.shape
        plv_matrix = np.zeros((n_trials, n_channels, n_channels))
        
        # Focus on alpha band (8-13 Hz) for motor imagery
        alpha_low = 8
        alpha_high = 13
        
        for trial_idx in range(n_trials):
            # Apply Hilbert transform to get instantaneous phase
            phases = np.zeros((n_channels, n_timepoints))
            
            for ch_idx in range(n_channels):
                signal = eeg_epochs[trial_idx, ch_idx, :]
                
                # Bandpass filter for alpha band
                filtered = self._bandpass_filter(signal, alpha_low, alpha_high)
                analytic_signal = hilbert(filtered)
                phases[ch_idx, :] = np.unwrap(np.angle(analytic_signal))
            
            # Compute PLV for each channel pair
            for ch1 in range(n_channels):
                for ch2 in range(n_channels):
                    if ch1 != ch2:
                        phase_diff = phases[ch1, :] - phases[ch2, :]
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        plv_matrix[trial_idx, ch1, ch2] = plv
        
        return plv_matrix
    
    def extract_erd_ers_features(self, eeg_epochs: np.ndarray, 
                                baseline_window: Tuple[float, float] = (-2.0, 0.0)) -> np.ndarray:
        """
        Extract Event-Related Desynchronization/Synchronization features
        
        Args:
            eeg_epochs: Shape (n_trials, n_channels, n_timepoints)
            baseline_window: Baseline time window in seconds
            
        Returns:
            ERD/ERS features shape (n_trials, n_channels, n_bands)
        """
        n_trials, n_channels, n_timepoints = eeg_epochs.shape
        
        # Convert time windows to sample indices
        baseline_start = int((baseline_window[0] + EEG_CONFIG['epoch_tmin']) * self.sampling_rate)
        baseline_end = int((baseline_window[1] + EEG_CONFIG['epoch_tmin']) * self.sampling_rate)
        
        # Frequency bands: alpha (8-13 Hz), beta (13-30 Hz)
        freq_bands = [(8, 13), (13, 30)]
        n_bands = len(freq_bands)
        
        erd_ers_features = np.zeros((n_trials, n_channels, n_bands))
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                signal = eeg_epochs[trial_idx, ch_idx, :]
                
                # Compute baseline power for each band
                baseline_power = []
                for band_low, band_high in freq_bands:
                    filtered_baseline = self._bandpass_filter(
                        signal[baseline_start:baseline_end], band_low, band_high
                    )
                    power = np.mean(filtered_baseline**2)
                    baseline_power.append(power)
                
                # Compute task power (2-4 seconds after event onset)
                task_start = int((2.0 + EEG_CONFIG['epoch_tmin']) * self.sampling_rate)
                task_end = int((4.0 + EEG_CONFIG['epoch_tmin']) * self.sampling_rate)
                
                for band_idx, (band_low, band_high) in enumerate(freq_bands):
                    filtered_task = self._bandpass_filter(
                        signal[task_start:task_end], band_low, band_high
                    )
                    task_power = np.mean(filtered_task**2)
                    
                    # ERD/ERS ratio (desynchronization if < 1, synchronization if > 1)
                    if baseline_power[band_idx] > 0:
                        erd_ers = task_power / baseline_power[band_idx]
                    else:
                        erd_ers = 1.0
                    
                    erd_ers_features[trial_idx, ch_idx, band_idx] = erd_ers
        
        return erd_ers_features
    
    def _bandpass_filter(self, signal: np.ndarray, low_freq: float, 
                        high_freq: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        return filtered


class fNIRSFeatureExtractor:
    """Feature extraction for fNIRS signals"""
    
    def __init__(self, sampling_rate: int = 10):
        self.sampling_rate = sampling_rate
        
    def extract_hemodynamic_features(self, hbo_epochs: np.ndarray, 
                                   hbr_epochs: np.ndarray) -> Dict:
        """
        Extract hemodynamic response features
        
        Args:
            hbo_epochs: Shape (n_trials, n_channels, n_timepoints)
            hbr_epochs: Shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            Dictionary containing hemodynamic features
        """
        n_trials, n_channels, n_timepoints = hbo_epochs.shape
        
        features = {
            'hbo_peak_amplitude': np.zeros((n_trials, n_channels)),
            'hbo_time_to_peak': np.zeros((n_trials, n_channels)),
            'hbo_area_under_curve': np.zeros((n_trials, n_channels)),
            'hbr_peak_amplitude': np.zeros((n_trials, n_channels)),
            'hbr_time_to_peak': np.zeros((n_trials, n_channels)),
            'hbr_area_under_curve': np.zeros((n_trials, n_channels)),
            'hbo_hbr_correlation': np.zeros((n_trials, n_channels))
        }
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                hbo_signal = hbo_epochs[trial_idx, ch_idx, :]
                hbr_signal = hbr_epochs[trial_idx, ch_idx, :]
                
                # Find peak and time to peak for HbO
                hbo_peak_idx = np.argmax(hbo_signal)
                hbo_peak = hbo_signal[hbo_peak_idx]
                hbo_time_peak = hbo_peak_idx / self.sampling_rate
                
                # Area under curve (trapezoidal integration)
                hbo_auc = np.trapz(hbo_signal, dx=1/self.sampling_rate)
                
                # Similar for HbR
                hbr_peak_idx = np.argmin(hbr_signal)  # HbR typically decreases
                hbr_peak = hbr_signal[hbr_peak_idx]
                hbr_time_peak = hbr_peak_idx / self.sampling_rate
                hbr_auc = np.trapz(hbr_signal, dx=1/self.sampling_rate)
                
                # Correlation between HbO and HbR
                correlation = np.corrcoef(hbo_signal, hbr_signal)[0, 1]
                
                # Store features
                features['hbo_peak_amplitude'][trial_idx, ch_idx] = hbo_peak
                features['hbo_time_to_peak'][trial_idx, ch_idx] = hbo_time_peak
                features['hbo_area_under_curve'][trial_idx, ch_idx] = hbo_auc
                features['hbr_peak_amplitude'][trial_idx, ch_idx] = hbr_peak
                features['hbr_time_to_peak'][trial_idx, ch_idx] = hbr_time_peak
                features['hbr_area_under_curve'][trial_idx, ch_idx] = hbr_auc
                features['hbo_hbr_correlation'][trial_idx, ch_idx] = correlation
        
        return features
    
    def extract_roi_features(self, hbo_epochs: np.ndarray, hbr_epochs: np.ndarray,
                           roi_definition: List[List[int]]) -> Dict:
        """
        Extract features from Regions of Interest (ROIs)
        
        Args:
            hbo_epochs: Shape (n_trials, n_channels, n_timepoints)
            hbr_epochs: Shape (n_trials, n_channels, n_timepoints)
            roi_definition: List of channel lists for each ROI
            
        Returns:
            Dictionary containing ROI features
        """
        n_trials, n_channels, n_timepoints = hbo_epochs.shape
        n_rois = len(roi_definition)
        
        features = {
            'roi_hbo_mean': np.zeros((n_trials, n_rois, n_timepoints)),
            'roi_hbo_std': np.zeros((n_trials, n_rois, n_timepoints)),
            'roi_hbr_mean': np.zeros((n_trials, n_rois, n_timepoints)),
            'roi_hbr_std': np.zeros((n_trials, n_rois, n_timepoints))
        }
        
        for trial_idx in range(n_trials):
            for roi_idx, roi_channels in enumerate(roi_definition):
                if len(roi_channels) == 0:
                    continue
                
                # Extract ROI channels
                roi_hbo = hbo_epochs[trial_idx, roi_channels, :]
                roi_hbr = hbr_epochs[trial_idx, roi_channels, :]
                
                # Compute ROI statistics
                features['roi_hbo_mean'][trial_idx, roi_idx, :] = np.mean(roi_hbo, axis=0)
                features['roi_hbo_std'][trial_idx, roi_idx, :] = np.std(roi_hbo, axis=0)
                features['roi_hbr_mean'][trial_idx, roi_idx, :] = np.mean(roi_hbr, axis=0)
                features['roi_hbr_std'][trial_idx, roi_idx, :] = np.std(roi_hbr, axis=0)
        
        return features
    
    def compute_activation_maps(self, hbo_epochs: np.ndarray, hbr_epochs: np.ndarray,
                              baseline_window: Tuple[int, int] = (0, 20)) -> np.ndarray:
        """
        Compute activation maps relative to baseline
        
        Args:
            hbo_epochs: Shape (n_trials, n_channels, n_timepoints)
            hbr_epochs: Shape (n_trials, n_channels, n_timepoints)
            baseline_window: Time window for baseline (start, end) in samples
            
        Returns:
            Activation maps shape (n_trials, n_channels, n_timepoints)
        """
        n_trials, n_channels, n_timepoints = hbo_epochs.shape
        
        # Compute baseline for each trial and channel
        baseline_start, baseline_end = baseline_window
        activation_maps = np.zeros_like(hbo_epochs)
        
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                # Baseline calculation
                hbo_baseline = np.mean(hbo_epochs[trial_idx, ch_idx, baseline_start:baseline_end])
                hbr_baseline = np.mean(hbr_epochs[trial_idx, ch_idx, baseline_start:baseline_end])
                
                # Compute activation (signal - baseline)
                hbo_activation = hbo_epochs[trial_idx, ch_idx, :] - hbo_baseline
                hbr_activation = hbr_epochs[trial_idx, ch_idx, :] - hbr_baseline
                
                # Combined activation (weighted sum)
                activation_maps[trial_idx, ch_idx, :] = (
                    0.7 * hbo_activation - 0.3 * hbr_activation
                )
        
        return activation_maps


class MultiModalFeatureFusion:
    """Feature fusion for multi-modal BCI data"""
    
    def __init__(self, fusion_method: str = 'attention'):
        self.fusion_method = fusion_method
    
    def early_fusion(self, eeg_features: Dict, fnirs_features: Dict) -> np.ndarray:
        """
        Early fusion: concatenate raw features
        
        Args:
            eeg_features: Dictionary of EEG features
            fnirs_features: Dictionary of fNIRS features
            
        Returns:
            Fused feature matrix
        """
        # Flatten and concatenate features
        eeg_flat = []
        for key, value in eeg_features.items():
            if key != 'complex_features':  # Skip complex features
                eeg_flat.append(value.reshape(value.shape[0], -1))
        
        fnirs_flat = []
        for key, value in fnirs_features.items():
            if 'roi' not in key:  # Skip time series ROI features for now
                fnirs_flat.append(value.reshape(value.shape[0], -1))
        
        eeg_concatenated = np.concatenate(eeg_flat, axis=1)
        fnirs_concatenated = np.concatenate(fnirs_flat, axis=1)
        
        # Final fusion
        fused_features = np.concatenate([eeg_concatenated, fnirs_concatenated], axis=1)
        
        return fused_features
    
    def late_fusion(self, eeg_predictions: np.ndarray, fnirs_predictions: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Late fusion: combine predictions from individual models
        
        Args:
            eeg_predictions: Predictions from EEG model
            fnirs_predictions: Predictions from fNIRS model
            weights: Optional weights for fusion
            
        Returns:
            Fused predictions
        """
        if weights is None:
            # Equal weights by default
            weights = np.array([0.5, 0.5])
        
        # Weighted average of predictions
        fused_predictions = weights[0] * eeg_predictions + weights[1] * fnirs_predictions
        
        return fused_predictions
    
    def attention_fusion(self, eeg_features: np.ndarray, fnirs_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attention-based fusion with learnable weights
        
        Args:
            eeg_features: EEG feature matrix
            fnirs_features: fNIRS feature matrix
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        n_samples = eeg_features.shape[0]
        n_eeg_features = eeg_features.shape[1]
        n_fnirs_features = fnirs_features.shape[1]
        
        # Simple attention mechanism (could be replaced with neural network)
        # Compute attention weights based on feature magnitude
        eeg_attention = np.mean(np.abs(eeg_features), axis=1, keepdims=True)
        fnirs_attention = np.mean(np.abs(fnirs_features), axis=1, keepdims=True)
        
        # Normalize attention weights
        total_attention = eeg_attention + fnirs_attention
        eeg_weights = eeg_attention / (total_attention + 1e-8)
        fnirs_weights = fnirs_attention / (total_attention + 1e-8)
        
        # Apply attention weights
        weighted_eeg = eeg_features * eeg_weights
        weighted_fnirs = fnirs_features * fnirs_weights
        
        # Fuse features
        fused_features = np.concatenate([weighted_eeg, weighted_fnirs], axis=1)
        
        attention_weights = np.concatenate([eeg_weights, fnirs_weights], axis=1)
        
        return fused_features, attention_weights


def create_feature_extractor(modality: str) -> object:
    """Factory function for feature extractors"""
    if modality.lower() == 'eeg':
        return EEGFeatureExtractor()
    elif modality.lower() == 'fnirs':
        return fNIRSFeatureExtractor()
    else:
        raise ValueError(f"Unknown modality: {modality}")


def check_module_availability():
    """Check if all required dependencies are available"""
    return MODULE_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeEEGFeatureExtractor:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (pywt, scipy) not available for feature extraction module")
        self.extractor = EEGFeatureExtractor(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.extractor, name)


class SafefNIRSFeatureExtractor:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (pywt, scipy) not available for feature extraction module")
        self.extractor = fNIRSFeatureExtractor(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.extractor, name)


class SafeMultiModalFeatureFusion:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (pywt, scipy) not available for feature extraction module")
        self.fusion = MultiModalFeatureFusion(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.fusion, name)