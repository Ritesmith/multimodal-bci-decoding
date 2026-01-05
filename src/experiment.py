"""
Experiment configuration and parameter tuning system for multi-modal BCI
"""
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# 延迟导入可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from sklearn.model_selection import ParameterGrid
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    ParameterGrid = None
    StratifiedKFold = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# 导入项目模块
from config import TRAINING_CONFIG, EVALUATION_CONFIG, RANDOM_SEED, RESULTS_DIR

try:
    from training import BCITrainer, CrossValidationTrainer, evaluate_model_performance
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    BCITrainer = None
    CrossValidationTrainer = None
    evaluate_model_performance = None

try:
    from models import MultiModalFusionModel, BaselineEEGModel, BaselinefNIRSModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    MultiModalFusionModel = None
    BaselineEEGModel = None
    BaselinefNIRSModel = None

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration management for experiments"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict:
        """Load default experiment configuration"""
        return {
            'experiment_name': f'bci_multimodal_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'dataset': {
                'type': 'bnci',
                'path': 'data/raw',
                'subjects': list(range(1, 41)),  # BNCI has 40 subjects
                'test_subjects': [1, 2, 3, 4],  # Hold-out subjects for final testing
                'preprocessing': {
                    'eeg': {
                        'notch_freq': 50,
                        'bandpass_low': 0.5,
                        'bandpass_high': 40,
                        'artifact_removal': True
                    },
                    'fnirs': {
                        'bandpass_low': 0.01,
                        'bandpass_high': 0.5,
                        'baseline_correction': True
                    }
                }
            },
            'models': {
                'multimodal': {
                    'enabled': True,
                    'fusion_method': 'attention',  # 'attention', 'concat', 'weighted'
                    'eeg_stream': {
                        'conv_filters': [32, 64, 128],
                        'lstm_units': [128, 64],
                        'dropout': 0.3
                    },
                    'fnirs_stream': {
                        'dense_units': [128, 64],
                        'dropout': 0.2
                    },
                    'attention_units': 64
                },
                'eeg_baseline': {
                    'enabled': True,
                    'conv_filters': [32, 64],
                    'lstm_units': [128, 64],
                    'dropout': 0.3
                },
                'fnirs_baseline': {
                    'enabled': True,
                    'dense_units': [128, 64],
                    'dropout': 0.2
                }
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'epochs': 100,
                'early_stopping': True,
                'patience': 10,
                'optimizer': 'AdamW',
                'weight_decay': 1e-4,
                'scheduler': 'CosineAnnealingLR'
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'itr'],
                'realtime_threshold_ms': 200
            },
            'hyperparameter_tuning': {
                'enabled': False,
                'method': 'optuna',  # 'grid', 'random', 'optuna'
                'n_trials': 100,
                'timeout_hours': 24
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Deep merge with default config
        self._deep_update(self.config, loaded_config)
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif config_path.suffix in ['.yml', '.yaml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


class HyperparameterTuner:
    """Hyperparameter tuning system"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.tuning_config = experiment_config.get('hyperparameter_tuning', {})
        self.results = []
        
    def grid_search(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, y: np.ndarray,
                   param_grid: Dict) -> List[Dict]:
        """
        Perform grid search hyperparameter tuning
        
        Args:
            param_grid: Dictionary of parameters to search
            X_eeg: EEG data
            X_fnirs: fNIRS data  
            y: Labels
            
        Returns:
            List of results for each parameter combination
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for grid search. Install with: pip install scikit-learn")
        
        logger.info("Starting grid search hyperparameter tuning...")
        
        results = []
        param_combinations = list(ParameterGrid(param_grid))
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Update training config with new parameters
                training_config = self.config.get('training', {}).copy()
                training_config.update(params)
                
                # Create and train model
                model = MultiModalFusionModel()
                trainer = BCITrainer(model, training_config)
                
                # Perform cross-validation
                cv_trainer = CrossValidationTrainer(
                    MultiModalFusionModel, {}, training_config
                )
                
                cv_results = cv_trainer.cross_validate(
                    X_eeg, X_fnirs, y, n_folds=self.config.get('evaluation.cv_folds', 5)
                )
                
                # Store results
                result = {
                    'params': params,
                    'mean_accuracy': cv_results['mean_accuracy'],
                    'std_accuracy': cv_results['std_accuracy'],
                    'mean_f1': cv_results['mean_f1'],
                    'cv_results': cv_results
                }
                
                results.append(result)
                logger.info(f"Results: Accuracy={result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                continue
        
        # Sort by accuracy
        results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        self.results = results
        
        return results
    
    def optuna_study(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, 
                    y: np.ndarray):
        """
        Perform Optuna hyperparameter optimization
        
        Args:
            X_eeg: EEG data
            X_fnirs: fNIRS data
            y: Labels
            
        Returns:
            Optuna study object
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'conv_filters': trial.suggest_categorical('conv_filters', [
                    [16, 32], [32, 64], [32, 64, 128], [64, 128, 256]
                ]),
                'lstm_units': trial.suggest_categorical('lstm_units', [
                    [64, 32], [128, 64], [256, 128]
                ])
            }
            
            try:
                # Create model with suggested parameters
                model_config = self.config.get('models.multimodal', {}).copy()
                model_config['eeg_stream'].update(params)
                model_config['fnirs_stream']['dropout'] = params['dropout']
                
                model = MultiModalFusionModel(model_config)
                
                # Training config
                training_config = self.config.get('training', {}).copy()
                training_config.update({
                    'learning_rate': params['learning_rate'],
                    'batch_size': params['batch_size'],
                    'weight_decay': params['weight_decay']
                })
                
                # Perform cross-validation (reduced folds for speed)
                cv_trainer = CrossValidationTrainer(
                    lambda: MultiModalFusionModel(model_config), 
                    {}, 
                    training_config
                )
                
                cv_results = cv_trainer.cross_validate(
                    X_eeg, X_fnirs, y, n_folds=3  # Reduced folds for speed
                )
                
                return cv_results['mean_accuracy']
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        
        n_trials = self.tuning_config.get('n_trials', 100)
        timeout_hours = self.tuning_config.get('timeout_hours', 24)
        timeout_seconds = timeout_hours * 3600
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        logger.info(f"Best accuracy: {study.best_trial.value:.4f}")
        
        return study
    
    def random_search(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, y: np.ndarray,
                     param_distributions: Dict, n_iterations: int = 50) -> List[Dict]:
        """
        Perform random search hyperparameter tuning
        
        Args:
            param_distributions: Dictionary of parameter distributions
            n_iterations: Number of random iterations
            X_eeg: EEG data
            X_fnirs: fNIRS data
            y: Labels
            
        Returns:
            List of results
        """
        logger.info(f"Starting random search with {n_iterations} iterations...")
        
        results = []
        
        for i in range(n_iterations):
            # Randomly sample parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, dict):
                    if distribution['type'] == 'uniform':
                        params[param_name] = np.random.uniform(
                            distribution['low'], distribution['high']
                        )
                    elif distribution['type'] == 'log_uniform':
                        params[param_name] = np.exp(
                            np.random.uniform(np.log(distribution['low']), 
                                            np.log(distribution['high']))
                        )
            
            logger.info(f"Iteration {i+1}/{n_iterations}: {params}")
            
            try:
                # Train and evaluate (similar to grid search)
                training_config = self.config.get('training', {}).copy()
                training_config.update(params)
                
                model = MultiModalFusionModel()
                trainer = BCITrainer(model, training_config)
                
                cv_trainer = CrossValidationTrainer(
                    MultiModalFusionModel, {}, training_config
                )
                
                cv_results = cv_trainer.cross_validate(
                    X_eeg, X_fnirs, y, n_folds=self.config.get('evaluation.cv_folds', 5)
                )
                
                result = {
                    'params': params,
                    'mean_accuracy': cv_results['mean_accuracy'],
                    'std_accuracy': cv_results['std_accuracy'],
                    'mean_f1': cv_results['mean_f1'],
                    'cv_results': cv_results
                }
                
                results.append(result)
                logger.info(f"Results: Accuracy={result['mean_accuracy']:.4f}±{result['std_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                continue
        
        results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        self.results = results
        
        return results


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_name = config.get('experiment_name')
        self.results_dir = RESULTS_DIR / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.data_loader = None
        self.models = {}
        self.results = {}
        
    def _setup_logging(self):
        """Setup experiment-specific logging"""
        log_file = self.results_dir / 'experiment.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess data"""
        logger.info("Loading and preprocessing data...")
        
        try:
            from data_loader import create_dataset_loader, DataPreprocessor, check_module_availability
            if not check_module_availability():
                raise ImportError("Data loader dependencies (mne, scipy) not available")
        except ImportError as e:
            logger.error(f"Failed to import data_loader module: {e}")
            raise ImportError("Data loader module not available. Please install required dependencies (mne, scipy)")
        
        dataset_type = self.config.get('dataset.type', 'bnci')
        data_path = self.config.get('dataset.path', 'data/raw')
        
        # Load data for multiple subjects
        all_eeg_data = []
        all_fnirs_data = []
        all_labels = []
        
        subjects = self.config.get('dataset.subjects', list(range(1, 41)))
        
        loader = create_dataset_loader(dataset_type, data_path)
        preprocessor = DataPreprocessor()
        
        for subject_id in subjects:
            try:
                logger.info(f"Loading subject {subject_id}")
                subject_data = loader.load_subject_data(subject_id)
                
                # Preprocess
                eeg_processed = preprocessor.preprocess_eeg(
                    subject_data['eeg'], subject_data['events']
                )
                fnirs_processed = preprocessor.preprocess_fnirs(
                    subject_data['fnirs'], subject_data['events']
                )
                
                # Synchronize
                synchronized = preprocessor.synchronize_modalities(
                    eeg_processed, fnirs_processed
                )
                
                all_eeg_data.append(synchronized['eeg'])
                all_fnirs_data.append(np.concatenate([
                    synchronized['hbo'], synchronized['hbr']
                ], axis=1))  # Combine HbO and HbR
                all_labels.append(synchronized['labels'])
                
            except Exception as e:
                logger.error(f"Error loading subject {subject_id}: {e}")
                continue
        
        # Concatenate all subjects
        X_eeg = np.concatenate(all_eeg_data, axis=0)
        X_fnirs = np.concatenate(all_fnirs_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Loaded data: EEG shape {X_eeg.shape}, fNIRS shape {X_fnirs.shape}, Labels shape {y.shape}")
        
        return X_eeg, X_fnirs, y
    
    def run_baseline_experiments(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, y: np.ndarray):
        """Run baseline experiments with single modalities"""
        logger.info("Running baseline experiments...")

        # Normalize labels to 0-based consecutive integers
        unique_labels = np.unique(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_normalized = np.array([label_mapping[label] for label in y])
        logger.info(f"Label mapping: {label_mapping}")

        baseline_results = {}

        # EEG baseline
        if self.config.get('models.eeg_baseline.enabled', True):
            logger.info("Training EEG baseline model...")

            training_config = self.config.get('training', {})

            # Use BCITrainer directly for baseline models
            from torch.utils.data import TensorDataset, DataLoader

            # Create dummy fNIRS data for compatibility
            dummy_fnirs = np.zeros((X_eeg.shape[0], 1, 1))

            # Create model

            eeg_model = BaselineEEGModel(
                X_eeg.shape[1], X_eeg.shape[2], len(unique_labels)
            )

            # Simple train/val split
            from sklearn.model_selection import train_test_split
            X_eeg_train, X_eeg_val, y_train, y_val = train_test_split(
                X_eeg, y_normalized, test_size=0.2, random_state=RANDOM_SEED, stratify=y_normalized
            )
            dummy_fnirs_train, dummy_fnirs_val = train_test_split(
                dummy_fnirs, test_size=0.2, random_state=RANDOM_SEED
            )
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_eeg_train),
                torch.FloatTensor(dummy_fnirs_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_eeg_val),
                torch.FloatTensor(dummy_fnirs_val),
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
            
            # Train model
            trainer = BCITrainer(eeg_model, training_config)
            trainer.train(train_loader, val_loader)
            
            # Evaluate
            results = evaluate_model_performance(eeg_model, val_loader)
            baseline_results['eeg'] = results
            logger.info(f"EEG baseline accuracy: {results['accuracy']:.4f}")
        
        # fNIRS baseline
        if self.config.get('models.fnirs_baseline.enabled', True):
            logger.info("Training fNIRS baseline model...")

            training_config = self.config.get('training', {})

            # Create dummy EEG data for compatibility
            dummy_eeg = np.zeros((X_fnirs.shape[0], 1, 1))

            # Create model
            fnirs_model = BaselinefNIRSModel(
                X_fnirs.shape[1], X_fnirs.shape[2], len(unique_labels)
            )

            # Simple train/val split
            from sklearn.model_selection import train_test_split
            X_fnirs_train, X_fnirs_val, y_train, y_val = train_test_split(
                X_fnirs, y_normalized, test_size=0.2, random_state=RANDOM_SEED, stratify=y_normalized
            )
            dummy_eeg_train, dummy_eeg_val = train_test_split(
                dummy_eeg, test_size=0.2, random_state=RANDOM_SEED
            )
            
            train_dataset = TensorDataset(
                torch.FloatTensor(dummy_eeg_train),
                torch.FloatTensor(X_fnirs_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(dummy_eeg_val),
                torch.FloatTensor(X_fnirs_val),
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
            
            # Train model
            trainer = BCITrainer(fnirs_model, training_config)
            trainer.train(train_loader, val_loader)
            
            # Evaluate
            results = evaluate_model_performance(fnirs_model, val_loader)
            baseline_results['fnirs'] = results
            logger.info(f"fNIRS baseline accuracy: {results['accuracy']:.4f}")

            # Create dummy EEG data for compatibility
            dummy_eeg = np.zeros((X_fnirs.shape[0], 1, 1))
            cv_results = cv_trainer.cross_validate(dummy_eeg, X_fnirs, y_normalized, n_folds=5)

            baseline_results['fnirs'] = cv_results
            logger.info(f"fNIRS baseline accuracy: {cv_results['mean_accuracy']:.4f}")

        self.results['baseline'] = baseline_results

    def run_multimodal_experiment(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, y: np.ndarray):
        """Run multi-modal experiment"""
        # Normalize labels to 0-based consecutive integers
        unique_labels = np.unique(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_normalized = np.array([label_mapping[label] for label in y])
        logger.info(f"Label mapping: {label_mapping}")
        logger.info("Running multi-modal experiment...")
        
        # Hyperparameter tuning
        if self.config.get('hyperparameter_tuning.enabled', False):
            tuner = HyperparameterTuner(self.config)

            method = self.config.get('hyperparameter_tuning.method', 'optuna')

            if method == 'grid':
                param_grid = {
                    'learning_rate': [1e-4, 1e-3, 1e-2],
                    'batch_size': [16, 32, 64],
                    'weight_decay': [1e-5, 1e-4, 1e-3]
                }
                tuning_results = tuner.grid_search(X_eeg, X_fnirs, y_normalized, param_grid)
            elif method == 'optuna':
                study = tuner.optuna_study(X_eeg, X_fnirs, y_normalized)
                tuning_results = study
            else:
                param_distributions = {
                    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
                    'batch_size': [16, 32, 64],
                    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-3}
                }
                n_iterations = self.config.get('hyperparameter_tuning.n_trials', 50)
                tuning_results = tuner.random_search(X_eeg, X_fnirs, y_normalized, param_distributions, n_iterations)

            self.results['tuning'] = tuning_results
        
        # Train final model with best parameters
        best_params = self.config.get('training', {})
        if 'tuning' in self.results:
            if isinstance(self.results['tuning'], list) and self.results['tuning']:
                best_params.update(self.results['tuning'][0]['params'])
            elif hasattr(self.results['tuning'], 'best_trial'):
                best_params.update(self.results['tuning'].best_trial.params)
        
        logger.info(f"Training final model with parameters: {best_params}")

        # Create config with actual input sizes
        model_config = {
            'eeg_stream': {
                'input_channels': X_eeg.shape[1],
                'input_length': X_eeg.shape[2],
                'conv_filters': [32, 64, 128],
                'conv_kernel': (3, 3),
                'lstm_units': [128, 64],
                'dropout': 0.3,
            },
            'fnirs_stream': {
                'input_channels': X_fnirs.shape[1],
                'input_length': X_fnirs.shape[2],
                'dense_units': [128, 64],
                'dropout': 0.2,
            },
            'fusion': {
                'output_classes': len(unique_labels),
                'fusion_method': 'attention',
                'attention_units': 64,
            }
        }

        model = MultiModalFusionModel(config=model_config)
        trainer = BCITrainer(model, best_params)

        cv_trainer = CrossValidationTrainer(
            lambda: MultiModalFusionModel(config=model_config), {}, best_params
        )

        cv_results = cv_trainer.cross_validate(X_eeg, X_fnirs, y_normalized, n_folds=5)
        
        self.results['multimodal'] = cv_results
        logger.info(f"Multi-modal accuracy: {cv_results['mean_accuracy']:.4f}")
    
    def run_experiments(self):
        """Run all experiments"""
        logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Load data
        X_eeg, X_fnirs, y = self.load_data()
        
        # Run baseline experiments
        self.run_baseline_experiments(X_eeg, X_fnirs, y)
        
        # Run multi-modal experiment
        self.run_multimodal_experiment(X_eeg, X_fnirs, y)
        
        # Save results
        self.save_results()
        
        logger.info("Experiments completed successfully!")
    
    def save_results(self):
        """Save experiment results"""
        results_file = self.results_dir / 'results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self) -> str:
        """Generate experiment report"""
        report = []
        report.append("# Multi-modal BCI Experiment Report\n")
        report.append(f"Experiment: {self.experiment_name}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if 'baseline' in self.results:
            report.append("## Baseline Results\n")
            for modality, results in self.results['baseline'].items():
                report.append(f"### {modality.upper()} Baseline")
                report.append(f"- Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
                report.append(f"- F1-Score: {results['mean_f1']:.4f}")
                report.append("")
        
        if 'multimodal' in self.results:
            report.append("## Multi-modal Results\n")
            results = self.results['multimodal']
            report.append(f"- Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            report.append(f"- F1-Score: {results['mean_f1']:.4f}")
            report.append("")
        
        if 'tuning' in self.results:
            report.append("## Hyperparameter Tuning Results\n")
            if isinstance(self.results['tuning'], list):
                report.append(f"Best parameters: {self.results['tuning'][0]['params']}")
                report.append(f"Best accuracy: {self.results['tuning'][0]['mean_accuracy']:.4f}")
        
        return "\n".join(report)


def create_default_experiment_config() -> str:
    """Create default experiment configuration file"""
    config = ExperimentConfig()
    config_path = "experiment_config.yaml"
    config.save_config(config_path)
    return config_path


if __name__ == "__main__":
    # Example usage
    config_path = create_default_experiment_config()
    
    # Load config
    config = ExperimentConfig(config_path)
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_experiments()
    
    # Generate report
    report = runner.generate_report()
    print(report)
    
    # Save report
    with open(runner.results_dir / 'report.md', 'w') as f:
        f.write(report)