"""
Evaluation and visualization module for multi-modal BCI decoding
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Safe imports for optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

try:
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    confusion_matrix = None
    classification_report = None
    TSNE = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

logger = logging.getLogger(__name__)

# Global flag to indicate if module is functional
MODULE_AVAILABLE = MATPLOTLIB_AVAILABLE and SKLEARN_AVAILABLE and PLOTLY_AVAILABLE and TORCH_AVAILABLE


if MODULE_AVAILABLE:
    class BCI_Evaluator:
        """Comprehensive evaluator for BCI models"""
        
        def __init__(self, class_names: Optional[List[str]] = None):
            self.class_names = class_names or ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        
        def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot confusion matrix
            
            Args:
                y_true: True labels
                y_pred: Predicted labels
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for plotting")
                
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names, ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def plot_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot classification metrics
            
            Args:
                y_true: True labels
                y_pred: Predicted labels
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
                raise ImportError("Required libraries not available for metrics plotting")
            
            report = classification_report(y_true, y_pred, 
                                       target_names=self.class_names,
                                       output_dict=True)
            
            # Extract metrics for plotting
            metrics = ['precision', 'recall', 'f1-score']
            class_metrics = {}
            
            for metric in metrics:
                class_metrics[metric] = [report[class_name][metric] 
                                      for class_name in self.class_names]
            
            # Create bar plot
            x = np.arange(len(self.class_names))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, class_metrics[metric], 
                       width, label=metric.capitalize())
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title('Classification Metrics by Class')
            ax.set_xticks(x + width)
            ax.set_xticklabels(self.class_names)
            ax.legend()
            ax.set_ylim(0, 1)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def plot_feature_importance(self, feature_names: List[str], 
                                 importance_scores: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot feature importance
            
            Args:
                feature_names: List of feature names
                importance_scores: Importance scores for features
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for plotting")
            
            # Sort features by importance
            sorted_idx = np.argsort(importance_scores)[::-1]
            sorted_names = [feature_names[i] for i in sorted_idx]
            sorted_scores = importance_scores[sorted_idx]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(sorted_names))
            
            ax.barh(y_pos, sorted_scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def plot_decision_boundaries(self, X_2d: np.ndarray, y: np.ndarray,
                                  model=None, save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot decision boundaries for 2D data
            
            Args:
                X_2d: 2D feature data
                y: Labels
                model: Trained model for decision boundary prediction
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for plotting")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data points
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                              cmap='viridis', alpha=0.7)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('Decision Boundaries (2D Projection)')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def plot_attention_weights(self, attention_weights: np.ndarray,
                               time_points: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot attention weights over time
            
            Args:
                attention_weights: Attention weights (n_samples, n_modalities)
                time_points: Time points for x-axis
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for plotting")
            
            n_samples, n_modalities = attention_weights.shape
            
            if time_points is None:
                time_points = np.arange(n_samples)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            modality_names = ['EEG', 'fNIRS']
            colors = ['blue', 'red']
            
            for i in range(n_modalities):
                ax.plot(time_points, attention_weights[:, i], 
                       label=modality_names[i], color=colors[i], linewidth=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Attention Weight')
            ax.set_title('Attention Weights Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def plot_brain_activation_heatmap(self, activation_data: np.ndarray,
                                      channel_names: List[str],
                                      save_path: Optional[str] = None) -> plt.Figure:
            """
            Plot brain activation heatmap
            
            Args:
                activation_data: Activation data (n_channels, n_timepoints)
                channel_names: Names of EEG channels
                save_path: Path to save the plot
                
            Returns:
                matplotlib figure
            """
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for plotting")
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create heatmap
            sns.heatmap(activation_data, 
                       yticklabels=channel_names,
                       cmap='viridis', 
                       ax=ax,
                       cbar_kws={'label': 'Activation'})
            
            ax.set_xlabel('Time Points')
            ax.set_ylabel('EEG Channels')
            ax.set_title('Brain Activation Heatmap')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        
        def create_interactive_dashboard(self, metrics: Dict, 
                                     save_path: Optional[str] = None):
            """
            Create interactive dashboard using Plotly
            
            Args:
                metrics: Dictionary containing evaluation metrics
                save_path: Path to save the HTML dashboard
                
            Returns:
                Plotly figure
            """
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly not available for interactive plots")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Add indicators for each metric
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('accuracy', 0),
                    title={'text': "Accuracy"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('precision', 0),
                    title={'text': "Precision"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('recall', 0),
                    title={'text': "Recall"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkorange"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics.get('f1', 0),
                    title={'text': "F1-Score"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkred"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Model Performance Dashboard")
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        
        def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   probabilities: np.ndarray,
                                   save_dir: str):
            """
            Generate comprehensive evaluation report
            
            Args:
                y_true: True labels
                y_pred: Predicted labels
                probabilities: Class probabilities
                save_dir: Directory to save all plots
            """
            if not MODULE_AVAILABLE:
                raise ImportError("Required dependencies not available for report generation")
            
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate various plots
            self.plot_confusion_matrix(y_true, y_pred, 
                                     os.path.join(save_dir, 'confusion_matrix.png'))
            
            self.plot_classification_metrics(y_true, y_pred,
                                         os.path.join(save_dir, 'classification_metrics.png'))
            
            # Generate text report
            report = classification_report(y_true, y_pred, 
                                       target_names=self.class_names)
            
            with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n")
                f.write(report)
            
            logger.info(f"Evaluation report saved to {save_dir}")

else:
    # Create placeholder class when dependencies are not available
    BCI_Evaluator = None


def check_module_availability():
    """Check if all required dependencies are available"""
    return MATPLOTLIB_AVAILABLE and SKLEARN_AVAILABLE and PLOTLY_AVAILABLE and TORCH_AVAILABLE


# Wrap classes to check availability before instantiation
class SafeBCI_Evaluator:
    def __init__(self, *args, **kwargs):
        if not check_module_availability():
            raise ImportError("Required dependencies (matplotlib, sklearn, plotly, torch) not available for evaluation module")
        self.evaluator = BCI_Evaluator(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.evaluator, name)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 probabilities: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Class probabilities
        
    Returns:
        Dictionary of metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn not available for metrics calculation")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    if len(np.unique(y_true)) <= 4:  # Reasonable number of classes
        metrics['precision_per_class'] = precision_score(y_true, y_pred, 
                                                     average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, 
                                               average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, 
                                          average=None, zero_division=0)
    
    # Additional metrics if probabilities are available
    if probabilities is not None and TORCH_AVAILABLE:
        try:
            metrics['auc'] = roc_auc_score(y_true, probabilities, 
                                         multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = None
    
    return metrics


if __name__ == "__main__":
    # Test evaluation functionality
    if check_module_availability():
        print("All dependencies available, evaluation module is functional")
        
        # Generate sample data for testing
        y_true = np.random.randint(0, 4, 100)
        y_pred = np.random.randint(0, 4, 100)
        
        evaluator = BCI_Evaluator(['Left', 'Right', 'Up', 'Down'])
        fig = evaluator.plot_confusion_matrix(y_true, y_pred)
        plt.show()
    else:
        print("Some dependencies missing, evaluation module has limited functionality")