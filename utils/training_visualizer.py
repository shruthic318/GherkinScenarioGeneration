import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import TrainerCallback

class TrainingMetricsLogger(TrainerCallback):
    """
    Callback to log training metrics during fine-tuning.
    Saves metrics to a JSON file and generates plots without TensorBoard.
    """
    def __init__(self, output_dir="training_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "training_metrics.json"
        self.metrics = {
            'train_loss': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # Log training metrics
        if 'loss' in logs and 'epoch' in logs:
            self.metrics['train_loss'].append(logs['loss'])
            self.metrics['learning_rate'].append(logs.get('learning_rate', 0))
            self.metrics['epoch'].append(logs['epoch'])
            self.metrics['step'].append(state.global_step)
            self._save_metrics()
            
    def on_train_end(self, args, state, control, **kwargs):
        self.plot_metrics()
        
    def _save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self):
        if not self.metrics['train_loss']:
            print("No training metrics to plot")
            return
            
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['step'], self.metrics['train_loss'], label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        loss_plot_path = self.output_dir / 'training_loss.png'
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Plot learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['step'], self.metrics['learning_rate'], label='Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        lr_plot_path = self.output_dir / 'learning_rate.png'
        plt.savefig(lr_plot_path)
        plt.close()
        
        print(f"Training metrics plots saved to {self.output_dir}")

def plot_from_json(metrics_file, output_dir=None):
    """
    Generate plots from a saved metrics JSON file.
    
    Args:
        metrics_file (str): Path to the metrics JSON file
        output_dir (str, optional): Directory to save the plots. Defaults to same as metrics file.
    """
    metrics_file = Path(metrics_file)
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    if not metrics['train_loss']:
        print("No training metrics to plot")
        return
        
    output_dir = Path(output_dir) if output_dir else metrics_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['step'], metrics['train_loss'], label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    loss_plot_path = output_dir / 'training_loss.png'
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot learning rate
    if 'learning_rate' in metrics and metrics['learning_rate']:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['step'], metrics['learning_rate'], label='Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        lr_plot_path = output_dir / 'learning_rate.png'
        plt.savefig(lr_plot_path)
        plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to the metrics JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save the plots (default: same as metrics file)')
    
    args = parser.parse_args()
    plot_from_json(args.metrics_file, args.output_dir)
