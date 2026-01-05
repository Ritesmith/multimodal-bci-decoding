"""
Main entry point for multi-modal BCI decoding project
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# 安全导入模块
try:
    from src.experiment import ExperimentRunner, ExperimentConfig
    EXPERIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Experiment module not available: {e}")
    EXPERIMENT_AVAILABLE = False
    ExperimentRunner = None
    ExperimentConfig = None

try:
    from src.data_loader import create_dataset_loader, DataPreprocessor
    DATALOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data loader module not available: {e}")
    DATALOADER_AVAILABLE = False
    create_dataset_loader = None
    DataPreprocessor = None

try:
    from src.feature_extraction import EEGFeatureExtractor, fNIRSFeatureExtractor, MultiModalFeatureFusion
    FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Feature extraction module not available: {e}")
    FEATURES_AVAILABLE = False
    EEGFeatureExtractor = None
    fNIRSFeatureExtractor = None
    MultiModalFeatureFusion = None

try:
    from src.models import MultiModalFusionModel, create_model, model_summary
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Models module not available: {e}")
    MODELS_AVAILABLE = False
    MultiModalFusionModel = None
    create_model = None
    model_summary = None

try:
    from src.training import BCITrainer, evaluate_model_performance
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Training module not available: {e}")
    TRAINING_AVAILABLE = False
    BCITrainer = None
    evaluate_model_performance = None

try:
    from src.evaluation import BCI_Evaluator
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Evaluation module not available: {e}")
    EVALUATION_AVAILABLE = False
    BCI_Evaluator = None

try:
    from src.realtime import RealTimeBCI, BCIDataSimulator
    REALTIME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Realtime module not available: {e}")
    REALTIME_AVAILABLE = False
    RealTimeBCI = None
    BCIDataSimulator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Multi-modal BCI Decoding System')
    parser.add_argument('--mode', choices=['train', 'eval', 'realtime', 'experiment'], 
                       default='experiment', help='Running mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--model-path', type=str, help='Path to saved model')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    
    args = parser.parse_args()
    
    if args.mode == 'experiment':
        run_experiment_mode(args)
    elif args.mode == 'train':
        run_train_mode(args)
    elif args.mode == 'eval':
        run_eval_mode(args)
    elif args.mode == 'realtime':
        run_realtime_mode(args)


def run_experiment_mode(args):
    """Run full experiment pipeline"""
    if not EXPERIMENT_AVAILABLE:
        print("Error: Experiment modules not available. Please install required dependencies.")
        print("Install with: pip install -r requirements.txt")
        return
    
    # Check for basic dependencies
    missing_deps = []
    if not DATALOADER_AVAILABLE:
        missing_deps.append("mne, scipy (for data loading)")
    if not MODELS_AVAILABLE:
        missing_deps.append("torch (for neural network models)")
    if not TRAINING_AVAILABLE:
        missing_deps.append("sklearn, matplotlib (for training and evaluation)")
    
    if missing_deps:
        print("Error: Some required dependencies are missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        print("Or install individual dependencies: pip install mne torch scikit-learn matplotlib")
        return
    
    logger.info("Running experiment mode...")
    
    # Load or create configuration
    if args.config:
        config = ExperimentConfig(args.config)
    else:
        config = ExperimentConfig()
        config_path = "experiment_config.yaml"
        config.save_config(config_path)
        logger.info(f"Created default config at {config_path}")
    
    # Override data path if provided
    if args.data_path:
        config._deep_update(config.config, {'dataset': {'path': args.data_path}})
    
    try:
        # Run experiments
        runner = ExperimentRunner(config)
        runner.run_experiments()
        
        # Generate and save report
        report = runner.generate_report()
        report_path = runner.results_dir / 'report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Experiment completed. Report saved to {report_path}")
    except ImportError as e:
        logger.error(f"Import error during experiment: {e}")
        print(f"\nError: {e}")
        print("Please install the missing dependencies and try again.")
        print("Install with: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        print(f"\nExperiment failed with error: {e}")
        print("Please check your data paths and configuration.")


def run_train_mode(args):
    """Train a single model"""
    if not (EXPERIMENT_AVAILABLE and DATALOADER_AVAILABLE and MODELS_AVAILABLE and TRAINING_AVAILABLE):
        print("Error: Training modules not available. Please install required dependencies.")
        print("Install with: pip install -r requirements.txt")
        return
    
    logger.info("Running training mode...")
    
    if not args.data_path:
        logger.error("Data path is required for training mode")
        return
    
    # Load configuration
    config = ExperimentConfig(args.config) if args.config else ExperimentConfig()
    
    # Load and preprocess data
    dataset_type = config.get('dataset.type', 'bnci')
    loader = create_dataset_loader(dataset_type, args.data_path)
    preprocessor = DataPreprocessor()
    
    # Load data for first subject as example
    subject_data = loader.load_subject_data(1)
    eeg_processed = preprocessor.preprocess_eeg(subject_data['eeg'], subject_data['events'])
    fnirs_processed = preprocessor.preprocess_fnirs(subject_data['fnirs'], subject_data['events'])
    synchronized = preprocessor.synchronize_modalities(eeg_processed, fnirs_processed)
    
    # Create model
    model = create_model('multimodal')
    
    # Print model summary
    model_summary(
        model,
        input_size_eeg=(2, synchronized['eeg'].shape[1], synchronized['eeg'].shape[2]),
        input_size_fnirs=(2, synchronized['hbo'].shape[1], synchronized['hbo'].shape[2])
    )
    
    # Train model
    from torch.utils.data import TensorDataset, DataLoader
    
    # Combine HbO and HbR for fNIRS
    fnirs_combined = np.concatenate([
        synchronized['hbo'], synchronized['hbr']
    ], axis=1)
    
    dataset = TensorDataset(
        torch.FloatTensor(synchronized['eeg']),
        torch.FloatTensor(fnirs_combined),
        torch.LongTensor(synchronized['labels'])
    )
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('training.batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('training.batch_size', 32), shuffle=False)
    
    # Train
    trainer = BCITrainer(model, config.get('training', {}))
    history = trainer.train(train_loader, val_loader, save_path='best_model.pth')
    
    logger.info("Training completed!")


def run_eval_mode(args):
    """Evaluate a trained model"""
    logger.info("Running evaluation mode...")
    
    if not args.model_path:
        logger.error("Model path is required for evaluation mode")
        return
    
    if not args.data_path:
        logger.error("Data path is required for evaluation mode")
        return
    
    # Load model
    model = create_model('multimodal')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data (similar to training mode)
    config = ExperimentConfig(args.config) if args.config else ExperimentConfig()
    dataset_type = config.get('dataset.type', 'bnci')
    loader = create_dataset_loader(dataset_type, args.data_path)
    preprocessor = DataPreprocessor()
    
    # Load test subject data
    test_subject = 4  # Use subject 4 for testing
    subject_data = loader.load_subject_data(test_subject)
    eeg_processed = preprocessor.preprocess_eeg(subject_data['eeg'], subject_data['events'])
    fnirs_processed = preprocessor.preprocess_fnirs(subject_data['fnirs'], subject_data['events'])
    synchronized = preprocessor.synchronize_modalities(eeg_processed, fnirs_processed)
    
    # Create test dataset
    fnirs_combined = np.concatenate([
        synchronized['hbo'], synchronized['hbr']
    ], axis=1)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(synchronized['eeg']),
        torch.FloatTensor(fnirs_combined),
        torch.LongTensor(synchronized['labels'])
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    results = evaluate_model_performance(model, test_loader)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"ITR: {results['itr']:.2f} bits/min")
    print(f"Avg Inference Time: {results['avg_inference_time_ms']:.2f} ms")
    
    # Generate visualizations
    evaluator = BCI_Evaluator()
    
    # Confusion matrix
    cm_fig = evaluator.plot_confusion_matrix(results['labels'], results['predictions'])
    cm_fig.savefig('confusion_matrix.png')
    
    # Decision boundaries (using extracted features)
    feature_extractor = EEGFeatureExtractor()
    eeg_features = feature_extractor.extract_time_frequency_features(synchronized['eeg'])
    features_flat = eeg_features['power'].reshape(len(synchronized['labels']), -1)
    
    tsne_fig = evaluator.plot_decision_boundaries(
        features_flat, synchronized['labels'], method='tsne'
    )
    tsne_fig.savefig('decision_boundaries.png')
    
    logger.info("Evaluation completed! Visualizations saved.")


def run_realtime_mode(args):
    """Run real-time BCI system"""
    logger.info("Running real-time mode...")
    
    if not args.model_path:
        logger.error("Model path is required for real-time mode")
        return
    
    # Load trained model
    model = create_model('multimodal')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create real-time system
    bci_system = RealTimeBCI(model, update_frequency=5.0)
    
    # Create simulator for testing
    simulator = BCIDataSimulator()
    
    # Prediction callback
    def on_prediction(result):
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"Class: {result['predicted_class']}, "
              f"Conf: {result['confidence']:.3f}, "
              f"Time: {result['inference_time_ms']:.1f}ms")
    
    # Start real-time prediction
    bci_system.start(callback=on_prediction)
    
    logger.info("Real-time BCI system started. Press Ctrl+C to stop.")
    
    try:
        # Simulate data streaming
        for i in range(200):  # Run for 20 seconds
            samples = simulator.get_samples(100)  # 100ms of data
            
            # Add data to buffers
            for j in range(samples['eeg'].shape[1]):
                bci_system.add_eeg_data(samples['eeg'][:, j:j+1])
            
            for j in range(samples['fnirs'].shape[1]):
                bci_system.add_fnirs_data(samples['fnirs'][:, j:j+1])
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Stopping real-time system...")
    
    finally:
        bci_system.stop()
        
        # Print performance stats
        stats = bci_system.get_performance_stats()
        print("\nReal-time Performance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    import time
    main()