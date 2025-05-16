#!/usr/bin/env python
# coding: utf8

"""Visualization utilities for audio source separation evaluation."""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import os


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, np.ndarray]],
    metric_name: str = "sdr",
    source_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_plot: bool = True,
) -> plt.Figure:
    """Plot comparison of metrics between different models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics dictionaries
        metric_name: Name of the metric to plot
        source_names: Names of the sources (e.g., ['vocals', 'accompaniment'])
        model_names: Names of the models to compare
        save_path: Path to save the figure
        figsize: Figure size
        show_plot: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    if not model_names:
        model_names = list(metrics_dict.keys())
    
    n_models = len(model_names)
    n_sources = metrics_dict[model_names[0]][metric_name].shape[0]
    
    if not source_names:
        source_names = [f"Source {i+1}" for i in range(n_sources)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Positions for the bars
    positions = np.arange(n_sources)
    width = 0.8 / n_models
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        model_metrics = metrics_dict[model_name]
        metric_values = model_metrics[metric_name]
        
        # Compute mean across any remaining dimensions (e.g., frames or frequency bands)
        if metric_values.ndim > 1:
            metric_values = np.nanmean(metric_values, axis=tuple(range(1, metric_values.ndim)))
        
        # Plot bars
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(positions + offset, metric_values, width, label=model_name)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.3,
                f"{height:.1f}",
                ha='center', va='bottom',
                fontsize=9
            )
    
    # Set labels and title
    ax.set_ylabel(f"{metric_name.upper()} (dB)")
    ax.set_title(f"{metric_name.upper()} comparison by source")
    ax.set_xticks(positions)
    ax.set_xticklabels(source_names)
    ax.legend()
    
    # Add grid and tight layout
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close figure
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_metrics_over_time(
    metrics_over_time: Dict[str, np.ndarray],
    source_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    show_plot: bool = True,
) -> plt.Figure:
    """Plot metrics over time for different sources.
    
    Args:
        metrics_over_time: Dictionary with metrics over time frames
        source_names: Names of the sources
        save_path: Path to save the figure
        figsize: Figure size
        show_plot: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    # Extract time information
    time_stamps = metrics_over_time.get("time", np.arange(metrics_over_time[list(metrics_over_time.keys())[0]].shape[-1]))
    
    # Get number of sources and metrics
    metrics = [m for m in metrics_over_time.keys() if m != "time"]
    n_metrics = len(metrics)
    n_sources = metrics_over_time[metrics[0]].shape[0]
    
    if not source_names:
        source_names = [f"Source {i+1}" for i in range(n_sources)]
    
    # Create figure
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    # Colors for different sources
    colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_values = metrics_over_time[metric]
        
        # Plot lines for each source
        for j in range(n_sources):
            ax.plot(
                time_stamps,
                metric_values[j],
                label=source_names[j],
                color=colors[j],
                marker='.',
                markersize=3,
                alpha=0.8
            )
        
        # Set labels and title
        ax.set_ylabel(f"{metric.upper()} (dB)")
        ax.set_title(f"{metric.upper()} over time")
        ax.grid(linestyle='--', alpha=0.7)
        
        # Add legend for the first subplot
        if i == 0:
            ax.legend()
    
    # Set common x-axis label
    axes[-1].set_xlabel("Time (seconds)")
    
    # Add tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close figure
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_frequency_dependent_metrics(
    freq_metrics: np.ndarray,
    frequencies: List[float],
    source_names: Optional[List[str]] = None,
    metric_name: str = "SDR",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    show_plot: bool = True,
) -> plt.Figure:
    """Plot frequency-dependent metrics.
    
    Args:
        freq_metrics: Metrics per frequency band, shape (n_sources, n_bands)
        frequencies: Center frequencies of each band
        source_names: Names of the sources
        metric_name: Name of the metric
        save_path: Path to save the figure
        figsize: Figure size
        show_plot: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    n_sources = freq_metrics.shape[0]
    
    if not source_names:
        source_names = [f"Source {i+1}" for i in range(n_sources)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors for different sources
    colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
    
    # Plot lines for each source
    for i in range(n_sources):
        ax.semilogx(
            frequencies,
            freq_metrics[i],
            label=source_names[i],
            color=colors[i],
            marker='o',
            markersize=6,
            linewidth=2,
            alpha=0.8
        )
    
    # Set labels and title
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"{metric_name} (dB)")
    ax.set_title(f"Frequency-dependent {metric_name}")
    
    # Set reasonable x-axis limits for audio
    ax.set_xlim([20, 20000])
    
    # Add grid and legend
    ax.grid(which='both', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add tight layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close figure
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def generate_evaluation_report(
    metrics_data: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    source_names: Optional[List[str]] = None,
) -> None:
    """Generate a comprehensive evaluation report with metrics and visualizations.
    
    Args:
        metrics_data: Dictionary mapping model names to metrics dictionaries
        output_dir: Directory to save the report
        source_names: Names of the sources
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine available metrics and models
    model_names = list(metrics_data.keys())
    metric_names = []
    for model_name in model_names:
        metric_names.extend(list(metrics_data[model_name].keys()))
    metric_names = sorted(list(set([m for m in metric_names if m != "time" and m != "permutation"])))
    
    # Determine number of sources
    n_sources = next(iter(metrics_data.values()))[metric_names[0]].shape[0]
    if not source_names:
        source_names = [f"Source {i+1}" for i in range(n_sources)]
    
    # Generate comparison plots for each metric
    for metric in metric_names:
        # Skip metrics that don't exist for all models
        if not all(metric in metrics_data[model] for model in model_names):
            continue
            
        save_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plot_metrics_comparison(
            metrics_data, 
            metric_name=metric,
            source_names=source_names,
            model_names=model_names,
            save_path=save_path,
            show_plot=False
        )
    
    # For each model, generate time-dependent plots if available
    for model_name in model_names:
        model_metrics = metrics_data[model_name]
        
        # Check if we have time-dependent data
        if "time" in model_metrics:
            time_metrics = {m: model_metrics[m] for m in metric_names if m in model_metrics}
            time_metrics["time"] = model_metrics["time"]
            
            save_path = os.path.join(output_dir, f"{model_name}_metrics_over_time.png")
            plot_metrics_over_time(
                time_metrics,
                source_names=source_names,
                save_path=save_path,
                show_plot=False
            )
    
    # Generate summary text report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("# Audio Source Separation Evaluation Summary\n\n")
        
        # Write overall metrics summary
        f.write("## Overall Performance Metrics\n\n")
        for metric in metric_names:
            f.write(f"### {metric.upper()}\n\n")
            f.write(f"{'Model':<20} " + " ".join([f"{src:>10}" for src in source_names]) + " Average\n")
            f.write("-" * (22 + 11 * len(source_names)) + "\n")
            
            for model_name in model_names:
                if metric not in metrics_data[model_name]:
                    continue
                    
                values = metrics_data[model_name][metric]
                if values.ndim > 1:
                    values = np.nanmean(values, axis=tuple(range(1, values.ndim)))
                
                avg = np.nanmean(values)
                f.write(f"{model_name:<20} " + " ".join([f"{v:>10.2f}" for v in values]) + f" {avg:>8.2f}\n")
            
            f.write("\n")
        
        # List generated visualizations
        f.write("## Generated Visualizations\n\n")
        for metric in metric_names:
            f.write(f"- {metric}_comparison.png: Comparison of {metric.upper()} across all models\n")
        
        for model_name in model_names:
            if "time" in metrics_data[model_name]:
                f.write(f"- {model_name}_metrics_over_time.png: Metrics over time for {model_name}\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("Add your conclusions and observations here based on the metrics results.\n") 