#!/usr/bin/env python
# coding: utf8

"""Main evaluation script for audio source separation models."""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

import argparse
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import glob
from datetime import datetime

from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from spleeter.evaluation.metrics.objective import (
    compute_metrics,
    frequency_dependent_sdr,
    compute_metrics_over_frames
)
from spleeter.evaluation.visualization.plots import (
    plot_metrics_comparison,
    plot_metrics_over_time,
    plot_frequency_dependent_metrics,
    generate_evaluation_report
)


def separate_audio(
    separator: Separator,
    audio_path: str,
    output_dir: str,
) -> Dict[str, np.ndarray]:
    """Separate audio file using the provided separator.
    
    Args:
        separator: Separator instance to use
        audio_path: Path to audio file
        output_dir: Directory to save separated audio
        
    Returns:
        Dictionary mapping source names to separated audio arrays
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform separation
    separator.separate_to_file(
        audio_path,
        output_dir,
        codec='wav',
        bitrate='256k',
        duration=None,
        offset=0.0
    )
    
    # Load separated audio files
    adapter = AudioAdapter.default()
    predictions = {}
    
    for source in separator._params["instrument_list"]:
        source_path = os.path.join(output_dir, os.path.basename(audio_path)[:-4], f"{source}.wav")
        if os.path.exists(source_path):
            audio_data, sample_rate = adapter.load(source_path, 0, None, separator._sample_rate)
            predictions[source] = audio_data
    
    return predictions


def load_reference_sources(
    reference_dir: str,
    instrument_list: List[str],
    target_sample_rate: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Load reference source files from directory.
    
    Args:
        reference_dir: Directory containing reference sources
        instrument_list: List of instrument names to load
        target_sample_rate: Target sample rate for loaded audio
        
    Returns:
        Tuple containing:
            - Dictionary mapping source names to audio arrays
            - Sample rate
    """
    adapter = AudioAdapter.default()
    references = {}
    
    for instrument in instrument_list:
        # Try different extensions
        for ext in ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']:
            path = os.path.join(reference_dir, f"{instrument}{ext}")
            if os.path.exists(path):
                audio_data, sample_rate = adapter.load(path, 0, None, target_sample_rate)
                references[instrument] = audio_data
                break
    
    return references, target_sample_rate


def evaluate_model(
    model: str,
    audio_path: str,
    reference_dir: str,
    output_dir: str,
    sample_rate: int = 44100,
    eval_metrics: List[str] = ["sdr", "sir", "sar", "si-sdr"],
    analyze_time_dependency: bool = True,
    analyze_freq_dependency: bool = True,
) -> Dict[str, np.ndarray]:
    """Evaluate a model on a given audio file with reference sources.
    
    Args:
        model: Model to evaluate (e.g., "2stems", "4stems")
        audio_path: Path to mixture audio file
        reference_dir: Directory containing reference source files
        output_dir: Directory to save evaluation results
        sample_rate: Sample rate for audio processing
        eval_metrics: List of metrics to compute
        analyze_time_dependency: Whether to analyze metrics over time
        analyze_freq_dependency: Whether to analyze frequency-dependent metrics
        
    Returns:
        Dictionary with computed metrics
    """
    # Create separator
    separator = Separator(model)
    
    # Create output directory
    model_output_dir = os.path.join(output_dir, f"model_{model}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Separate audio
    predictions = separate_audio(
        separator, 
        audio_path, 
        os.path.join(model_output_dir, "separated")
    )
    
    # Load reference sources
    references, _ = load_reference_sources(
        reference_dir,
        separator._params["instrument_list"],
        sample_rate
    )
    
    # Check that we have the same instruments in predictions and references
    common_instruments = list(set(predictions.keys()) & set(references.keys()))
    if not common_instruments:
        raise ValueError(
            f"No common instruments found between predictions and references. "
            f"Predictions: {list(predictions.keys())}. "
            f"References: {list(references.keys())}."
        )
    
    # Ensure same length for all sources (truncate to the shortest)
    min_length = min(
        min([predictions[instr].shape[0] for instr in common_instruments]),
        min([references[instr].shape[0] for instr in common_instruments])
    )
    
    # Prepare arrays for metrics computation
    reference_arrays = np.array([references[instr][:min_length] for instr in common_instruments])
    prediction_arrays = np.array([predictions[instr][:min_length] for instr in common_instruments])
    
    # Compute global metrics
    global_metrics = compute_metrics(
        reference_arrays,
        prediction_arrays,
        sample_rate,
        metrics=eval_metrics,
        compute_permutation=True
    )
    
    # Initialize results dictionary
    all_metrics = {}
    for metric, values in global_metrics.items():
        if metric != "permutation":
            all_metrics[metric] = values
    
    # Analyze time-dependency
    if analyze_time_dependency:
        frame_metrics = compute_metrics_over_frames(
            reference_arrays,
            prediction_arrays,
            sample_rate,
            frame_length=1.0,
            hop_length=0.5,
            metrics=eval_metrics
        )
        
        # Plot time-dependent metrics
        plot_metrics_over_time(
            frame_metrics,
            source_names=common_instruments,
            save_path=os.path.join(model_output_dir, "metrics_over_time.png"),
            show_plot=False
        )
        
        # Add time-dependent metrics to results
        for metric in eval_metrics:
            all_metrics[f"{metric}_over_time"] = frame_metrics[metric]
        all_metrics["time"] = frame_metrics["time"]
    
    # Analyze frequency-dependency
    if analyze_freq_dependency and "sdr" in eval_metrics:
        freq_sdr, band_freqs = frequency_dependent_sdr(
            reference_arrays,
            prediction_arrays,
            sample_rate,
            n_bands=8
        )
        
        # Plot frequency-dependent SDR
        plot_frequency_dependent_metrics(
            freq_sdr,
            band_freqs,
            source_names=common_instruments,
            save_path=os.path.join(model_output_dir, "frequency_dependent_sdr.png"),
            show_plot=False
        )
        
        # Add frequency-dependent metrics to results
        all_metrics["freq_sdr"] = freq_sdr
        all_metrics["band_freqs"] = np.array(band_freqs)
    
    # Save metrics to JSON file
    with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in all_metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=2)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio source separation models")
    parser.add_argument("--models", nargs="+", required=True, 
                        help="Models to evaluate (e.g., 2stems, 4stems)")
    parser.add_argument("--audio", required=True, 
                        help="Path to mixture audio file")
    parser.add_argument("--reference-dir", required=True, 
                        help="Directory containing reference source files")
    parser.add_argument("--output-dir", default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--sample-rate", type=int, default=44100, 
                        help="Sample rate for audio processing")
    parser.add_argument("--metrics", nargs="+", 
                        default=["sdr", "sir", "sar", "si-sdr"], 
                        help="Metrics to compute")
    parser.add_argument("--no-time", action="store_true", 
                        help="Skip time-dependent analysis")
    parser.add_argument("--no-freq", action="store_true", 
                        help="Skip frequency-dependent analysis")
    
    args = parser.parse_args()
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each model
    all_model_metrics = {}
    for model in args.models:
        print(f"Evaluating model: {model}")
        metrics = evaluate_model(
            model,
            args.audio,
            args.reference_dir,
            output_dir,
            sample_rate=args.sample_rate,
            eval_metrics=args.metrics,
            analyze_time_dependency=not args.no_time,
            analyze_freq_dependency=not args.no_freq
        )
        all_model_metrics[model] = metrics
        print(f"Done evaluating {model}")
    
    # Generate comparative evaluation report
    if len(args.models) > 1:
        print("Generating comparative evaluation report")
        # Extract just the global metrics for the report
        report_metrics = {}
        for model, metrics in all_model_metrics.items():
            report_metrics[model] = {
                metric: values for metric, values in metrics.items()
                if metric in args.metrics or metric == "time"
            }
        
        # Determine source names from the first model
        first_model = next(iter(all_model_metrics.keys()))
        source_names = args.models[0].split("stems")
        
        generate_evaluation_report(
            report_metrics,
            os.path.join(output_dir, "comparative_report"),
            source_names=None  # Will be auto-determined based on data shape
        )
    
    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 