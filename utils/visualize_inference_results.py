import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import seaborn as sns


def parse_csv_file(file_path: Path) -> List[dict]:
    """Parse a CSV inference result file and return metrics."""
    print(f"  Parsing file: {file_path}")
    if not file_path.exists():
        print("  File does not exist!")
        return []

    try:
        # Handle base model files: inference_results_<size>_<model>.csv
        base_match = re.match(r'inference_results_(\d+)_(\w+)\.csv$', file_path.name)
        if base_match:
            sample_size = int(base_match.group(1))
            model_name = base_match.group(2)
            model_type = 'lora' if sample_size == 1000 else 'base'
            print(f"  Found {'LoRA' if sample_size == 1000 else 'Base'} model: {model_name}, size: {sample_size}")
        else:
            # Handle DoRA/IA3 files: inference_results_<type>_<model>.csv
            model_match = re.match(r'inference_results_(dora|ia3)_(\w+)\.csv$', file_path.name, re.IGNORECASE)
            if model_match:
                model_type = model_match.group(1).lower()
                model_name = model_match.group(2)
                sample_size = 1000  # For DoRA/IA3, we'll use 1000 as the sample size
                print(f"  Found {model_type.upper()} model: {model_name}")
            else:
                print(f"  File doesn't match expected patterns: {file_path.name}")
                return []
        
        # Read the CSV file
        print(f"  Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"  CSV columns: {df.columns.tolist()}")
        print(f"  Model values: {df['model'].unique().tolist() if 'model' in df.columns else 'No model column found'}")
        
        results = []
        
        # Process each model in the CSV
        for _, row in df.iterrows():
            if row['model'] == 'fine-tuned' or (model_type == 'base' and row['model'] == 'base'):
                result = {
                    'sample_size': sample_size,
                    'model': model_name,
                    'model_type': model_type,
                    'base_model': model_name,
                    'BLEU Score': row.get('bleu', 0),
                    'ROUGE-2': row.get('rouge2', 0),
                    'ROUGE-L': row.get('rougel', row.get('rougeL', 0)),
                    'BERTScore (F1)': row.get('bertscore_f1', 0)
                }
                results.append(result)
        
        print(f"  Processed {len(results)} rows")
        return results
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_all_results(results_dir: Path) -> List[dict]:
    """Load all inference results from the directory."""
    all_results = []
    
    # Look for CSV files
    pattern = 'inference_results_*.csv'
    print(f"Looking for files matching: {pattern} in {results_dir}")
    found_files = list(results_dir.glob(pattern))
    print(f"Found {len(found_files)} files matching pattern")
    
    for file_path in found_files:
        print(f"\nProcessing file: {file_path.name}")
        if 'hybridlora' in file_path.name:
            print("Skipping hybrid lora file")
            continue
            
        entries = parse_csv_file(file_path)
        if entries:
            print(f"Found {len(entries)} valid entries in {file_path.name}")
            all_results.extend(entries)
    
    print(f"\nTotal results loaded: {len(all_results)}")
    return all_results


def plot_metric_comparison(df: pd.DataFrame, metric: str, output_dir: Path):
    """Generate comparison plots with model types on x-axis and models in legend."""
    if df.empty:
        return
    
    # Create a copy of the dataframe for plotting
    plot_df = df.copy()
    
    # Map model types to display names
    type_display = {
        'base': 'Base',
        'lora': 'LoRA',
        'dora': 'DoRA',
        'ia3': 'IA3'
    }
    
    # Create a new column for display names
    plot_df['model_type_display'] = plot_df['model_type'].map(type_display)
    
    # Define a consistent color palette for model types using display names as keys
    model_type_palette = {
        'Base': '#1f77b4',   # muted blue
        'LoRA': '#ff7f0e',   # safety orange
        'DoRA': '#2ca02c',   # cooked asparagus green
        'IA3': '#d62728'     # brick red
    }
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Create the plot with consistent colors
    ax = sns.barplot(
        data=plot_df,
        x='base_model',
        y=metric,
        hue='model_type_display',
        palette=model_type_palette,
        order=sorted(plot_df['base_model'].unique()),
        hue_order=['Base', 'LoRA', 'DoRA', 'IA3']
    )
    
    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(
            container, 
            fmt='%.3f',  # Format to 3 decimal places
            padding=3,
            fontsize=9,
            label_type='edge'
        )
    
    # Customize the plot
    plt.title(f'{metric} Comparison by Model and Type', fontsize=14)
    plt.xlabel('Model', labelpad=10, fontsize=12)
    plt.ylabel(metric, labelpad=10, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Adjust y-axis limits to make room for the value labels
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)  # Add 10% more space at the top
    
    # Adjust legend
    plt.legend(
        title='Model Type',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=10
    )
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Save the figure
    metric_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_file = output_dir / f'{metric_name}_comparison.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_scaling_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots showing how metrics change with sample size for each model."""
    if df.empty:
        return
    
    # Filter to include both base and lora models with sample sizes 100, 400, 1000
    plot_df = df[df['model_type'].isin(['base', 'lora'])].copy()
    
    # For 1000-sample models, we need to ensure we're using the lora model_type
    plot_df.loc[plot_df['sample_size'] == 1000, 'model_type'] = 'base'
    
    # Define metrics to plot
    metrics = ['BLEU Score', 'ROUGE-2', 'ROUGE-L', 'BERTScore (F1)']
    
    # Define a color palette for sample sizes
    sample_size_palette = {
        100: '#1f77b4',   # blue
        400: '#ff7f0e',   # orange
        1000: '#2ca02c'   # green
    }
    
    # Create a plot for each metric
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        # Create the plot
        ax = sns.barplot(
            data=plot_df,
            x='base_model',
            y=metric,
            hue='sample_size',
            palette=sample_size_palette,
            order=sorted(plot_df['base_model'].unique()),
            hue_order=[100, 400, 1000],
            dodge=True  # Ensure bars are properly grouped
        )
        
        # Add value labels on top of each bar
        for container in ax.containers:
            ax.bar_label(
                container, 
                fmt='%.3f',  # Format to 3 decimal places
                padding=3,
                fontsize=9,
                label_type='edge'
            )
        
        # Customize the plot
        plt.title(f'{metric} Comparison by Model and Sample Size', fontsize=14)
        plt.xlabel('Model', labelpad=10, fontsize=12)
        plt.ylabel(metric, labelpad=10, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Adjust y-axis limits to make room for the value labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.1)  # Add 10% more space at the top
        
        # Adjust legend
        plt.legend(
            title='Sample Size',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=10
        )
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Save the figure
        metric_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
        output_file = output_dir / f'{metric_name}_datascaling_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Set up paths
    current_dir = Path(__file__).parent
    results_dir = current_dir.parent 
    output_dir = current_dir.parent / 'comparison_plots/inference_results'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No valid results found. Exiting.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Generate individual metric comparison plots
    metric_columns = [col for col in df.columns if col not in 
                     ['sample_size', 'model', 'model_type', 'base_model']]
    
    for metric in metric_columns:
        plot_metric_comparison(df, metric, output_dir)
    
    # Generate data scaling comparison plots
    plot_data_scaling_comparison(df, output_dir)
    
    print("\nAll visualizations have been generated successfully!")

if __name__ == "__main__":
    main()
