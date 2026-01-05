"""
Real-time inference interface for multi-modal BCI decoding
"""
import numpy as np
from collections import deque
import threading
import time
from typing import Dict, List, Optional, Callable
import logging

from config import REALTIME_CONFIG, DEVICE

# Safe imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from models import MultiModalFusionModel, BaselineEEGModel, BaselinefNIRSModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    MultiModalFusionModel = None
    BaselineEEGModel = None
    BaselinefNIRSModel = None

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = TORCH_AVAILABLE and MODELS_AVAILABLE


if MODULE_AVAILABLE:
    class RealTimeBCI:
        """
        Real-time BCI inference system
        """
        
        def __init__(self, model: nn.Module, buffer_size: int = 1000, 
                     update_frequency: float = 5.0):
            """
            Initialize real-time BCI system
            
            Args:
                model: Trained neural network model
                buffer_size: Size of the data buffer
                update_frequency: Frequency of predictions in Hz
            """
            self.model = model
            self.buffer_size = buffer_size
            self.update_frequency = update_frequency
            self.device = torch.device(DEVICE)
            
            # Model evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            # Data buffers
            self.eeg_buffer = deque(maxlen=buffer_size)
            self.fnirs_buffer = deque(maxlen=buffer_size)
            
            # Prediction variables
            self.current_prediction = None
            self.prediction_history = deque(maxlen=100)
            self.is_running = False
            
            # Threading
            self.prediction_thread = None
            self.data_lock = threading.Lock()
            
            # Timing
            self.update_interval = 1.0 / update_frequency
            self.last_update_time = time.time()
            
        def add_eeg_data(self, eeg_sample: np.ndarray):
            """
            Add EEG sample to buffer
            
            Args:
                eeg_sample: EEG data sample
            """
            with self.data_lock:
                self.eeg_buffer.append(eeg_sample.copy())
        
        def add_fnirs_data(self, fnirs_sample: np.ndarray):
            """
            Add fNIRS sample to buffer
            
            Args:
                fnirs_sample: fNIRS data sample
            """
            with self.data_lock:
                self.fnirs_buffer.append(fnirs_sample.copy())
        
        def add_multimodal_data(self, eeg_sample: np.ndarray, fnirs_sample: np.ndarray):
            """
            Add multi-modal samples to buffers
            
            Args:
                eeg_sample: EEG data sample
                fnirs_sample: fNIRS data sample
            """
            self.add_eeg_data(eeg_sample)
            self.add_fnirs_data(fnirs_sample)
        
        def _prepare_inference_data(self) -> Optional[Dict]:
            """
            Prepare data for inference
            
            Returns:
                Dictionary with prepared tensors or None if not enough data
            """
            with self.data_lock:
                if len(self.eeg_buffer) == 0 or len(self.fnirs_buffer) == 0:
                    return None
                
                # Get most recent samples
                eeg_data = np.array(list(self.eeg_buffer))
                fnirs_data = np.array(list(self.fnirs_buffer))
                
                # Ensure correct shapes for model
                # Assume model expects: (batch_size, channels, time)
                if len(eeg_data.shape) == 1:
                    eeg_data = eeg_data.reshape(1, -1)  # (1, features)
                if len(fnirs_data.shape) == 1:
                    fnirs_data = fnirs_data.reshape(1, -1)  # (1, features)
                
                return {
                    'eeg': torch.FloatTensor(eeg_data).to(self.device),
                    'fnirs': torch.FloatTensor(fnirs_data).to(self.device)
                }
        
        def _predict(self):
            """
            Make prediction on current buffer data
            """
            data = self._prepare_inference_data()
            if data is None:
                return
            
            try:
                with torch.no_grad():
                    if hasattr(self.model, 'eeg_encoder'):  # Multi-modal model
                        outputs, attention_weights = self.model(data['eeg'], data['fnirs'])
                    else:  # Single modality model
                        outputs = self.model(data['eeg'])
                        attention_weights = None
                    
                    # Get prediction
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, dim=1)
                    
                    # Update prediction
                    self.current_prediction = {
                        'class_id': predicted_class.item(),
                        'confidence': confidence.item(),
                        'probabilities': probabilities.cpu().numpy()[0],
                        'attention_weights': attention_weights.cpu().numpy()[0] if attention_weights is not None else None,
                        'timestamp': time.time()
                    }
                    
                    self.prediction_history.append(self.current_prediction.copy())
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
        
        def start(self):
            """Start real-time prediction"""
            if self.is_running:
                logger.warning("Real-time BCI is already running")
                return
            
            self.is_running = True
            self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
            self.prediction_thread.start()
            logger.info("Real-time BCI started")
        
        def stop(self):
            """Stop real-time prediction"""
            if not self.is_running:
                logger.warning("Real-time BCI is not running")
                return
            
            self.is_running = False
            if self.prediction_thread:
                self.prediction_thread.join(timeout=1.0)
            logger.info("Real-time BCI stopped")
        
        def _prediction_loop(self):
            """Main prediction loop running in separate thread"""
            while self.is_running:
                current_time = time.time()
                
                if current_time - self.last_update_time >= self.update_interval:
                    self._predict()
                    self.last_update_time = current_time
                
                # Sleep to prevent excessive CPU usage
                time.sleep(0.01)  # 10ms
        
        def get_current_prediction(self) -> Optional[Dict]:
            """
            Get the most recent prediction
            
            Returns:
                Prediction dictionary or None if no prediction available
            """
            return self.current_prediction
        
        def get_prediction_history(self, n_predictions: int = None) -> List[Dict]:
            """
            Get prediction history
            
            Args:
                n_predictions: Number of recent predictions to return
                
            Returns:
                List of prediction dictionaries
            """
            if n_predictions is None:
                return list(self.prediction_history)
            else:
                return list(self.prediction_history)[-n_predictions:]
        
        def get_statistics(self, window_size: int = 50) -> Dict:
            """
            Get prediction statistics over a window
            
            Args:
                window_size: Size of the analysis window
                
            Returns:
                Statistics dictionary
            """
            recent_predictions = self.get_prediction_history(window_size)
            
            if not recent_predictions:
                return {}
            
            # Extract class predictions
            class_ids = [p['class_id'] for p in recent_predictions]
            confidences = [p['confidence'] for p in recent_predictions]
            
            # Calculate statistics
            unique_classes, counts = np.unique(class_ids, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            
            return {
                'most_common_class': most_common_class,
                'class_counts': dict(zip(unique_classes, counts)),
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'total_predictions': len(recent_predictions),
                'prediction_frequency': len(recent_predictions) / max(1, (time.time() - recent_predictions[0]['timestamp']))
            }

    class BCIDataSimulator:
        """
        Simulate BCI data for testing real-time system
        """
        
        def __init__(self, eeg_channels: int = 64, fnirs_channels: int = 10,
                     sample_rate: float = 250.0, noise_level: float = 0.1):
            """
            Initialize simulator
            
            Args:
                eeg_channels: Number of EEG channels
                fnirs_channels: Number of fNIRS channels
                sample_rate: Sampling rate in Hz
                noise_level: Level of noise to add
            """
            self.eeg_channels = eeg_channels
            self.fnirs_channels = fnirs_channels
            self.sample_rate = sample_rate
            self.noise_level = noise_level
            
            # State variables
            self.time = 0.0
            self.current_class = 0
            self.class_duration = 5.0  # seconds
            self.last_class_change = 0.0
            
            # Signal parameters
            self.frequencies = np.random.uniform(1, 40, eeg_channels)
            self.phases = np.random.uniform(0, 2*np.pi, eeg_channels)
            
        def generate_sample(self) -> tuple[np.ndarray, np.ndarray]:
            """
            Generate one sample of multi-modal data
            
            Returns:
                Tuple of (eeg_sample, fnirs_sample)
            """
            # Update time
            self.time += 1.0 / self.sample_rate
            
            # Change class periodically
            if self.time - self.last_class_change > self.class_duration:
                self.current_class = np.random.randint(0, 4)
                self.last_class_change = self.time
                # Update signal parameters for new class
                self.frequencies = np.random.uniform(1, 40, self.eeg_channels)
                self.phases = np.random.uniform(0, 2*np.pi, self.eeg_channels)
            
            # Generate EEG signal
            t = self.time
            eeg_signal = np.zeros(self.eeg_channels)
            
            for i in range(self.eeg_channels):
                # Class-specific frequency components
                if self.current_class == 0:
                    eeg_signal[i] = np.sin(2 * np.pi * 10 * t + self.phases[i])
                elif self.current_class == 1:
                    eeg_signal[i] = np.sin(2 * np.pi * 15 * t + self.phases[i])
                elif self.current_class == 2:
                    eeg_signal[i] = np.sin(2 * np.pi * 20 * t + self.phases[i])
                else:
                    eeg_signal[i] = np.sin(2 * np.pi * 25 * t + self.phases[i])
                
                # Add noise
                eeg_signal[i] += np.random.normal(0, self.noise_level)
            
            # Generate fNIRS signal (slower changes)
            fnirs_signal = np.zeros(self.fnirs_channels)
            for i in range(self.fnirs_channels):
                # Simulate hemodynamic response
                base_level = 0.5 + 0.3 * self.current_class / 4.0
                fnirs_signal[i] = base_level + 0.1 * np.sin(2 * np.pi * 0.5 * t + i * np.pi / self.fnirs_channels)
                fnirs_signal[i] += np.random.normal(0, self.noise_level * 0.5)
            
            return eeg_signal, fnirs_signal
        
        def generate_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
            """
            Generate a batch of samples
            
            Args:
                n_samples: Number of samples to generate
                
            Returns:
                Tuple of (eeg_batch, fnirs_batch)
            """
            eeg_batch = []
            fnirs_batch = []
            
            for _ in range(n_samples):
                eeg_sample, fnirs_sample = self.generate_sample()
                eeg_batch.append(eeg_sample)
                fnirs_batch.append(fnirs_sample)
            
            return np.array(eeg_batch), np.array(fnirs_batch)
        
        def stream_to_realtime_bci(self, bci_system: RealTimeBCI, duration: float):
            """
            Stream simulated data to real-time BCI system
            
            Args:
                bci_system: RealTimeBCI instance
                duration: Duration in seconds
            """
            logger.info(f"Starting data stream for {duration} seconds")
            
            n_samples = int(duration * REALTIME_CONFIG['sample_rate'])
            start_time = time.time()
            
            bci_system.start()
            
            try:
                for i in range(n_samples):
                    eeg_sample, fnirs_sample = self.generate_sample()
                    bci_system.add_multimodal_data(eeg_sample, fnirs_sample)
                    
                    # Control timing
                    elapsed = time.time() - start_time
                    expected_time = i / REALTIME_CONFIG['sample_rate']
                    if elapsed < expected_time:
                        time.sleep(expected_time - elapsed)
                    
                    # Print progress
                    if i % (n_samples // 10) == 0:
                        logger.info(f"Progress: {100 * i / n_samples:.1f}%")
                
            finally:
                bci_system.stop()
                logger.info("Data streaming completed")

else:
    # Create placeholder classes when dependencies are not available
    RealTimeBCI = None
    BCIDataSimulator = None


def check_module_availability():
    """Check if all required dependencies are available"""
    return TORCH_AVAILABLE and MODELS_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeRealTimeBCI:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch, models) not available for realtime module")
        self.bci = RealTimeBCI(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.bci, name)


class SafeBCIDataSimulator:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch, models) not available for realtime module")
        self.simulator = BCIDataSimulator(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.simulator, name)


def test_realtime_system():
    """Test real-time BCI system functionality"""
    if not check_module_availability():
        print("Required dependencies not available for real-time testing")
        return
    
    try:
        # Create a simple model for testing
        model = MultiModalFusionModel()
        
        # Create real-time BCI system
        bci_system = RealTimeBCI(model, buffer_size=100, update_frequency=10.0)
        
        # Create simulator
        simulator = BCIDataSimulator(eeg_channels=64, fnirs_channels=10)
        
        # Test streaming
        simulator.stream_to_realtime_bci(bci_system, duration=5.0)
        
        # Get statistics
        stats = bci_system.get_statistics()
        print("Real-time BCI Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Real-time test failed: {e}")


if __name__ == "__main__":
    test_realtime_system()