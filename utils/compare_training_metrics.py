import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


def load_metrics(model_dir: Path, sample_size: str) -> Dict[str, Any]:
    """Load metrics from a model's training directory."""
    metrics_file = model_dir / sample_size / 'training_metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Add step numbers if not present
    if 'step' not in metrics or not metrics['step']:
        metrics['step'] = list(range(len(metrics['train_loss'])))
    for key, value in metrics.items():
        if isinstance(value, (list, tuple)):
            metrics[key] = np.array(value)
    
    return metrics

def plot_metrics(metrics_dict: Dict[str, Dict[str, Any]], metric_name: str, 
                title: str, output_file: Path, sample_size: str, log_scale: bool = False,
                figsize=(12, 6)):
    """Plot metrics across different models."""
    plt.figure(figsize=figsize)
    
    for model_name, metrics in metrics_dict.items():
        if metric_name in metrics:
            # Clean up model name for legend
            clean_name = model_name.replace('_', ' ').replace('finetune', '').strip().title()
            plt.plot(metrics[metric_name], label=clean_name, linewidth=2)
    
    plt.title(f'{title} (Sample Size: {sample_size})', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def find_metrics_files(base_dir: str, model_name: str, sample_size: str) -> List[str]:
    """Find all metrics files for a given model and sample size."""
    metrics_dir = Path(base_dir) / model_name / sample_size
    if not metrics_dir.exists():
        return []
    return list(metrics_dir.glob("metrics.json"))

def load_metrics_for_model(model_dir: Path, model_identifier: str) -> Dict[str, Any]:
    """Load metrics for a model.
    
    Args:
        model_dir: Directory containing the model's metrics
        model_identifier: Either 'dora', 'ia3', or a sample size (e.g., '100', '400', '1000')
    """
    metrics_path = model_dir / model_identifier / 'training_metrics.json'
    return load_metrics_file(metrics_path)

def load_metrics_file(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def generate_comparison_plots(base_dir: str, models: List[str], sample_sizes: List[str], output_base_dir: Path):
    """Generate comparison plots for multiple models and sample sizes."""
    base_path = Path(base_dir)
    
    # Create output directories with training_metrics_comparison subdirectory
    output_base_dir = output_base_dir / 'training_metrics_comparison'
    dora_dir = output_base_dir / 'dora'
    ia3_dir = output_base_dir / 'ia3'
    dora_dir.mkdir(parents=True, exist_ok=True)
    ia3_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics storage
    all_base_metrics = {size: {} for size in sample_sizes}
    all_dora_metrics = {size: {} for size in sample_sizes}
    all_ia3_metrics = {size: {} for size in sample_sizes}
    
    # Collect all metrics
    for sample_size in sample_sizes:

        
        for model in models:
            model_dir = base_path / model
            
            # Load base model metrics
            metrics = load_metrics_for_model(model_dir, sample_size)
            all_base_metrics[sample_size][model] = metrics

            
            # Load DoRA metrics
            dora_metrics = load_metrics_for_model(model_dir, 'dora')
            all_dora_metrics[sample_size][f"{model}_dora"] = dora_metrics

            
            # Load IA3 metrics
            ia3_metrics = load_metrics_for_model(model_dir, 'ia3')
            all_ia3_metrics[sample_size][f"{model}_ia3"] = ia3_metrics

    
    # Generate base model comparison plots (one per sample size)
    for sample_size, metrics in all_base_metrics.items():
        output_dir = output_base_dir / f'sample_{sample_size}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_combined_metrics(
            metrics,
            f'Base Models - Training Loss (Sample Size: {sample_size})',
            output_dir / 'training_loss.png'
        )

    
    # Generate DoRA plot
    plot_combined_metrics(
        all_dora_metrics['1000'],
        'DoRA Models - Training Loss (Sample Size: 1000)',
        dora_dir / 'training_loss.png'
    )

    
    # Generate IA3 plot 
    plot_combined_metrics(
        all_ia3_metrics['1000'],
        'IA3 Models - Training Loss (Sample Size: 1000)',
        ia3_dir / 'training_loss.png'
    )


def plot_individual_metrics(metrics_dict: Dict[str, Dict[str, Any]], 
                          sample_size: str, output_dir: Path):
    """Plot individual metrics for all models."""
    # Only plot training loss, no learning rate
    plot_metrics(
        metrics_dict, 
        'train_loss', 
        'Training Loss Comparison', 
        output_dir / 'training_loss.png',
        sample_size
    )

def plot_combined_metrics(metrics_dict: Dict[str, Dict[str, Any]], 
                        title: str, 
                        output_path: Path
                        ):
    """Plot combined training loss for multiple models with the same sample size."""
    plt.figure(figsize=(12, 6))
    
    for model_name, metrics in metrics_dict.items():
        if 'train_loss' in metrics:
            # Clean up the model name for the legend
            display_name = model_name.split('/')[-1].replace('_', ' ').title()
            plt.plot(metrics['train_loss'], label=display_name)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_combined_metrics_all_sizes(metrics_dict: Dict[str, Dict[str, Any]], 
                                 title: str, 
                                 output_path: Path,
                                 model_type: str):
    """Plot combined training loss for all sample sizes in one plot."""
    plt.figure(figsize=(12, 6))
    
    # Define a color palette for different models
    colors = plt.cm.tab10.colors
    model_colors = {}
    
    # First pass: collect all unique models
    all_models = set()
    for size_metrics in metrics_dict.values():
        all_models.update(size_metrics.keys())
    
    # Assign colors to models
    for i, model in enumerate(sorted(all_models)):
        model_colors[model] = colors[i % len(colors)]
    
    # Plot each model's metrics for each sample size
    for sample_size, size_metrics in metrics_dict.items():
        if not size_metrics:
            continue
            
        for model_name, metrics in size_metrics.items():
            if 'train_loss' in metrics:
                display_name = f"{model_name.replace(f'_{model_type}', '')} ({sample_size})"
                plt.plot(
                    metrics['train_loss'], 
                    label=display_name,
                    color=model_colors[model_name],
                    linestyle='--' if '100' in sample_size else ('-.' if '400' in sample_size else '-')
                )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare training metrics across models and sample sizes')
    parser.add_argument('--base-dir', type=str, required=True, 
                       help='Base directory containing model metrics')
    parser.add_argument('--models', type=str, nargs='+', required=True, 
                       help='List of model names to compare')
    parser.add_argument('--output-dir', type=str, default='comparison_plots', 
                       help='Directory to save comparison plots')
    args = parser.parse_args()
    
    # Define sample sizes to process
    sample_sizes = ['100', '400', '1000']
    
    # Create main output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for all sample sizes
    generate_comparison_plots(args.base_dir, args.models, sample_sizes, output_dir)


if __name__ == "__main__":
    main()