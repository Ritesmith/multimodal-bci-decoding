"""
Training and validation routines for multi-modal BCI models
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

# Safe imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StratifiedKFold = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    roc_auc_score = None

try:
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    tqdm = None

from config import TRAINING_CONFIG, EVALUATION_CONFIG, RANDOM_SEED, DEVICE
try:
    from models import MultiModalFusionModel, BaselineEEGModel, BaselinefNIRSModel, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    MultiModalFusionModel = None
    BaselineEEGModel = None
    BaselinefNIRSModel = None

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = TORCH_AVAILABLE and SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE


if MODULE_AVAILABLE:
    class BCITrainer:
        """Trainer class for BCI models"""
        
        def __init__(self, model: nn.Module, config: Optional[Dict] = None):
            self.model = model
            self.config = config or TRAINING_CONFIG
            self.device = torch.device(DEVICE)
            self.model.to(self.device)
            
            # Setup optimizer
            optimizer_name = self.config['optimizer']
            learning_rate = self.config['learning_rate']
            weight_decay = self.config['weight_decay']
            
            if optimizer_name == 'AdamW':
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_name == 'Adam':
                self.optimizer = optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            else:
                self.optimizer = optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
            
            # Setup loss function
            self.criterion = nn.CrossEntropyLoss()
            
            # Setup scheduler
            scheduler_name = self.config.get('scheduler', None)
            if scheduler_name == 'CosineAnnealingLR':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config['epochs']
                )
            elif scheduler_name == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=30, gamma=0.1
                )
            else:
                self.scheduler = None
            
            # Training history
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
            
        def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                if len(batch) == 3:  # Multi-modal: eeg, fnirs, labels
                    eeg, fnirs, labels = batch
                    eeg, fnirs, labels = eeg.to(self.device), fnirs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(eeg, fnirs)
                    if isinstance(outputs, tuple):
                        outputs, attention = outputs
                else:  # Single modality: data, labels
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        outputs, attention = outputs
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_samples:.4f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_samples
            return avg_loss, accuracy
        
        def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
            """Validate for one epoch"""
            self.model.eval()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc="Validation")
                for batch in pbar:
                    if len(batch) == 3:  # Multi-modal: eeg, fnirs, labels
                        eeg, fnirs, labels = batch
                        eeg, fnirs, labels = eeg.to(self.device), fnirs.to(self.device), labels.to(self.device)

                        outputs = self.model(eeg, fnirs)
                        if isinstance(outputs, tuple):
                            outputs, attention = outputs
                    else:  # Single modality: data, labels
                        data, labels = batch
                        data, labels = data.to(self.device), labels.to(self.device)
                        
                        outputs = self.model(data)
                    
                    loss = self.criterion(outputs, labels)
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{correct_predictions/total_samples:.4f}'
                    })
            
            avg_loss = total_loss / len(val_loader)
            accuracy = correct_predictions / total_samples
            return avg_loss, accuracy
        
        def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
                 save_path: str = None, early_stopping_patience: int = 10) -> Dict:
            """
            Train model
            
            Args:
                train_loader: Training data loader
                val_loader: Validation data loader (optional)
                save_path: Path to save best model (optional)
                early_stopping_patience: Patience for early stopping
                
            Returns:
                Training history dictionary
            """
            epochs = self.config['epochs']
            best_val_accuracy = 0.0
            patience_counter = 0
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                
                # Training
                train_loss, train_accuracy = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_accuracy)
                
                # Validation
                if val_loader:
                    val_loss, val_accuracy = self.validate_epoch(val_loader)
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_accuracy)
                    
                    logger.info(
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                    )
                    
                    # Early stopping and model saving
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        if save_path:
                            torch.save(self.model.state_dict(), save_path)
                            logger.info(f"Best model saved to {save_path}")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {patience_counter} epochs")
                            break
                else:
                    logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                
                # Learning rate scheduler
                if self.scheduler:
                    self.scheduler.step()
            
            # Load best model if validation was used
            if val_loader and save_path and torch.cuda.is_available():
                self.model.load_state_dict(torch.load(save_path))
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'best_val_accuracy': best_val_accuracy
            }

        def plot_training_history(self, save_path: str = None):
            """Plot training and validation curves"""
            if not MATPLOTLIB_AVAILABLE:
                print("Matplotlib not available, skipping plots")
                return
                
            plt.figure(figsize=(15, 5))
            
            # Loss curves
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Train Loss')
            if self.val_losses:
                plt.plot(self.val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            # Accuracy curves
            plt.subplot(1, 2, 2)
            plt.plot(self.train_accuracies, label='Train Accuracy')
            if self.val_accuracies:
                plt.plot(self.val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    class CrossValidationTrainer:
        """Cross-validation trainer for model evaluation"""
        
        def __init__(self, model_class, model_params: Dict, config: Dict):
            self.model_class = model_class
            self.model_params = model_params
            self.config = config
            self.cv_results = []
            
        def cross_validate(self, X_eeg: np.ndarray, X_fnirs: np.ndarray, 
                          y: np.ndarray, n_folds: int = 5) -> Dict:
            """
            Perform cross-validation
            
            Args:
                X_eeg: EEG data
                X_fnirs: fNIRS data  
                y: Labels
                n_folds: Number of cross-validation folds
                
            Returns:
                Cross-validation results
            """
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
            
            fold_accuracies = []
            fold_f1_scores = []
            fold_precisions = []
            fold_recalls = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_eeg, y)):
                logger.info(f"Fold {fold + 1}/{n_folds}")
                
                # Split data
                X_eeg_train, X_eeg_val = X_eeg[train_idx], X_eeg[val_idx]
                X_fnirs_train, X_fnirs_val = X_fnirs[train_idx], X_fnirs[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                model = self.model_class(**self.model_params)
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_eeg_train), 
                    torch.FloatTensor(X_fnirs_train), 
                    torch.LongTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_eeg_val), 
                    torch.FloatTensor(X_fnirs_val), 
                    torch.LongTensor(y_val)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
                
                # Train model
                trainer = BCITrainer(model, self.config)
                history = trainer.train(train_loader, val_loader)
                
                # Evaluate on validation set
                model.eval()
                all_predictions = []
                all_labels = []
                
                with torch.no_grad():
                    for eeg_batch, fnirs_batch, labels_batch in val_loader:
                        eeg_batch = eeg_batch.to(trainer.device)
                        fnirs_batch = fnirs_batch.to(trainer.device)
                        
                        outputs, _ = model(eeg_batch, fnirs_batch)
                        _, predicted = torch.max(outputs, 1)
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels_batch.numpy())
                
                # Calculate metrics
                accuracy = accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
                recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
                f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
                
                fold_accuracies.append(accuracy)
                fold_f1_scores.append(f1)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                
                logger.info(f"Fold {fold + 1} - Acc: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Store results
            self.cv_results = {
                'fold_accuracies': fold_accuracies,
                'fold_f1_scores': fold_f1_scores,
                'fold_precisions': fold_precisions,
                'fold_recalls': fold_recalls,
                'mean_accuracy': np.mean(fold_accuracies),
                'std_accuracy': np.std(fold_accuracies),
                'mean_f1': np.mean(fold_f1_scores),
                'std_f1': np.std(fold_f1_scores)
            }
            
            logger.info(f"CV Results - Mean Acc: {self.cv_results['mean_accuracy']:.4f} ± {self.cv_results['std_accuracy']:.4f}")
            
            return self.cv_results

        def plot_cv_results(self, save_path: str = None):
            """Plot cross-validation results"""
            if not MATPLOTLIB_AVAILABLE or not self.cv_results:
                print("Matplotlib not available or no CV results, skipping plots")
                return
            
            metrics = ['accuracy', 'f1', 'precision', 'recall']
            values = [
                self.cv_results['fold_accuracies'],
                self.cv_results['fold_f1_scores'],
                self.cv_results['fold_precisions'],
                self.cv_results['fold_recalls']
            ]
            means = [
                self.cv_results['mean_accuracy'],
                self.cv_results['mean_f1'],
                np.mean(self.cv_results['fold_precisions']),
                np.mean(self.cv_results['fold_recalls'])
            ]
            stds = [
                self.cv_results['std_accuracy'],
                self.cv_results['std_f1'],
                np.std(self.cv_results['fold_precisions']),
                np.std(self.cv_results['fold_recalls'])
            ]
            
            plt.figure(figsize=(12, 6))
            
            # Box plot
            plt.subplot(1, 2, 1)
            plt.boxplot(values, labels=metrics)
            plt.title('Cross-Validation Performance Distribution')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            # Bar plot with error bars
            plt.subplot(1, 2, 2)
            x_pos = np.arange(len(metrics))
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xticks(x_pos, metrics)
            plt.title('Mean Cross-Validation Performance')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

else:
    # Create placeholder classes when dependencies are not available
    BCITrainer = None
    CrossValidationTrainer = None


def check_module_availability():
    """Check if all required dependencies are available"""
    return TORCH_AVAILABLE and SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeBCITrainer:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch, sklearn, matplotlib) not available for training module")
        self.trainer = BCITrainer(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.trainer, name)


class SafeCrossValidationTrainer:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (torch, sklearn, matplotlib) not available for training module")
        self.trainer = CrossValidationTrainer(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.trainer, name)


def evaluate_model_performance(model, test_loader, 
                             device: str = DEVICE) -> Dict:
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for computation
        
    Returns:
        Performance metrics
    """
    if not MODULE_AVAILABLE:
        raise ImportError("Required dependencies not available for model evaluation")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for batch in pbar:
            if len(batch) == 3:  # Multi-modal: eeg, fnirs, labels
                eeg, fnirs, labels = batch
                eeg, fnirs, labels = eeg.to(device), fnirs.to(device), labels.to(device)

                outputs = model(eeg, fnirs)
                # Check if model returns attention weights as well
                if isinstance(outputs, tuple):
                    outputs, attention = outputs
            else:  # Single modality: data, labels
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                # Check if model returns attention weights as well
                if isinstance(outputs, tuple):
                    outputs, attention = outputs
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probabilities,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }


def save_training_history(history: Dict, save_path: str):
    """Save training history to file"""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.floating, np.integer)):
            history_serializable[key] = [float(x) for x in value]
        elif isinstance(value, (np.floating, np.integer)):
            history_serializable[key] = float(value)
        else:
            history_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    logger.info(f"Training history saved to {save_path}")


if __name__ == "__main__":
    # Test training functionality
    if check_module_availability():
        print("All dependencies available, training module is functional")
    else:
        print("Some dependencies missing, training module has limited functionality")