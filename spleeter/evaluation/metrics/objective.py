#!/usr/bin/env python
# coding: utf8

"""Objective metrics for audio source separation evaluation."""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
from fast_bss_eval import bss_eval_sources, si_bss_eval_sources

AudioType = Union[np.ndarray, torch.Tensor]


def compute_metrics(
    references: AudioType,
    estimates: AudioType,
    sample_rate: int,
    metrics: List[str] = ["sdr", "sir", "sar", "si-sdr"],
    compute_permutation: bool = True,
    filter_length: int = 512,
    zero_mean: bool = False,
    use_cg_iter: Optional[int] = 10,
    load_diag: Optional[float] = 1e-10,
    clamp_db: Optional[float] = None,
) -> Dict[str, AudioType]:
    """Compute source separation metrics between estimated and reference sources.

    Args:
        references: Reference sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        estimates: Estimated sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        sample_rate: Sample rate of audio signals
        metrics: List of metrics to compute
        compute_permutation: Whether to compute the best permutation of estimates
        filter_length: Length of the distortion filter allowed for BSSEval metrics
        zero_mean: Whether to zero-mean the signals before metric computation
        use_cg_iter: Number of iterations for conjugate gradient solver (faster)
        load_diag: Small value added to diagonal for stability
        clamp_db: Clamp metrics to (-clamp_db, clamp_db) range in dB

    Returns:
        Dictionary with computed metrics
    """
    # Convert inputs to numpy if they are torch tensors
    if isinstance(references, torch.Tensor):
        references_np = references.detach().cpu().numpy()
    else:
        references_np = references

    if isinstance(estimates, torch.Tensor):
        estimates_np = estimates.detach().cpu().numpy()
    else:
        estimates_np = estimates

    results = {}

    # BSS Eval metrics
    if any(m in metrics for m in ["sdr", "sir", "sar"]):
        sdr, sir, sar, perm = bss_eval_sources(
            references_np,
            estimates_np,
            compute_permutation=compute_permutation,
            filter_length=filter_length,
            zero_mean=zero_mean,
            use_cg_iter=use_cg_iter,
            load_diag=load_diag,
            clamp_db=clamp_db,
        )
        if "sdr" in metrics:
            results["sdr"] = sdr
        if "sir" in metrics:
            results["sir"] = sir
        if "sar" in metrics:
            results["sar"] = sar
        if compute_permutation:
            results["permutation"] = perm

    # Scale-invariant BSS Eval metrics
    if any(m in metrics for m in ["si-sdr", "si-sir", "si-sar"]):
        si_sdr, si_sir, si_sar, si_perm = si_bss_eval_sources(
            references_np,
            estimates_np,
            compute_permutation=compute_permutation,
            zero_mean=zero_mean,
            clamp_db=clamp_db,
        )
        if "si-sdr" in metrics:
            results["si-sdr"] = si_sdr
        if "si-sir" in metrics:
            results["si-sir"] = si_sir
        if "si-sar" in metrics:
            results["si-sar"] = si_sar
        if compute_permutation and "permutation" not in results:
            results["permutation"] = si_perm

    return results


def frequency_dependent_sdr(
    references: AudioType,
    estimates: AudioType,
    sample_rate: int,
    n_bands: int = 4,
    filter_length: int = 512,
    zero_mean: bool = False,
    use_cg_iter: Optional[int] = 10,
    load_diag: Optional[float] = 1e-10,
) -> Tuple[AudioType, List[float]]:
    """Compute frequency-dependent SDR.

    Args:
        references: Reference sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        estimates: Estimated sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        sample_rate: Sample rate of audio signals
        n_bands: Number of frequency bands
        filter_length: Length of the distortion filter allowed
        zero_mean: Whether to zero-mean the signals before metric computation
        use_cg_iter: Number of iterations for conjugate gradient solver
        load_diag: Small value added to diagonal for stability

    Returns:
        Tuple containing:
            - SDR values per band per source, shape (n_sources, n_bands)
            - Band frequencies (center frequencies of each band)
    """
    if isinstance(references, torch.Tensor):
        references_np = references.detach().cpu().numpy()
    else:
        references_np = references

    if isinstance(estimates, torch.Tensor):
        estimates_np = estimates.detach().cpu().numpy()
    else:
        estimates_np = estimates
    
    # Check dimensions
    if references_np.ndim == 2:
        references_np = references_np[np.newaxis, ...]
        estimates_np = estimates_np[np.newaxis, ...]
    
    batch_size, n_sources, n_samples = references_np.shape
    
    # Create logarithmically spaced frequency bands
    nyquist = sample_rate // 2
    min_freq = 20  # Hz, lowest audible frequency
    band_edges = np.logspace(np.log10(min_freq), np.log10(nyquist), n_bands + 1)
    band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])
    
    # Initialize result array
    freq_sdrs = np.zeros((batch_size, n_sources, n_bands))
    
    # Compute SDR for each frequency band
    for b in range(n_bands):
        # Bandpass filter the signals
        low_freq = band_edges[b]
        high_freq = band_edges[b + 1]
        
        # Normalize frequencies to [0, 1] for scipy's filter design
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Apply bandpass filtering (using FFT)
        from scipy.signal import firwin, freqz, lfilter
        
        # Design FIR bandpass filter
        filter_order = min(int(sample_rate / low_freq * 4), 1000)  # Adapt filter order based on frequency
        if filter_order % 2 == 0:
            filter_order += 1  # Ensure odd filter order
            
        bandpass_filter = firwin(
            filter_order, 
            [low_norm, high_norm], 
            pass_zero=False, 
            fs=2.0  # Normalized frequency
        )
        
        # Apply filter to references and estimates
        references_band = np.zeros_like(references_np)
        estimates_band = np.zeros_like(estimates_np)
        
        for i in range(batch_size):
            for j in range(n_sources):
                references_band[i, j] = lfilter(bandpass_filter, 1.0, references_np[i, j])
                estimates_band[i, j] = lfilter(bandpass_filter, 1.0, estimates_np[i, j])
        
        # Compute SDR for this band
        sdr, _, _, _ = bss_eval_sources(
            references_band,
            estimates_band,
            compute_permutation=False,  # Use original source order since we're analyzing per band
            filter_length=filter_length,
            zero_mean=zero_mean,
            use_cg_iter=use_cg_iter,
            load_diag=load_diag,
        )
        
        freq_sdrs[:, :, b] = sdr
    
    # Remove batch dimension if there was none originally
    if batch_size == 1:
        freq_sdrs = freq_sdrs[0]
    
    return freq_sdrs, band_centers.tolist()


def compute_metrics_over_frames(
    references: AudioType,
    estimates: AudioType,
    sample_rate: int,
    frame_length: float = 1.0,  # in seconds
    hop_length: float = 0.5,    # in seconds
    metrics: List[str] = ["sdr"],
    filter_length: int = 512,
    zero_mean: bool = False,
    use_cg_iter: Optional[int] = 10,
    load_diag: Optional[float] = 1e-10,
) -> Dict[str, np.ndarray]:
    """Compute metrics over overlapping frames for temporal consistency analysis.
    
    Args:
        references: Reference sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        estimates: Estimated sources, shape (n_sources, n_samples) or (batch, n_sources, n_samples)
        sample_rate: Sample rate of audio signals
        frame_length: Frame length in seconds
        hop_length: Hop length in seconds
        metrics: List of metrics to compute
        filter_length: Length of the distortion filter allowed
        zero_mean: Whether to zero-mean the signals before metric computation
        use_cg_iter: Number of iterations for conjugate gradient solver
        load_diag: Small value added to diagonal for stability
        
    Returns:
        Dictionary with computed metrics over frames
    """
    if isinstance(references, torch.Tensor):
        references_np = references.detach().cpu().numpy()
    else:
        references_np = references

    if isinstance(estimates, torch.Tensor):
        estimates_np = estimates.detach().cpu().numpy()
    else:
        estimates_np = estimates
        
    # Check dimensions
    if references_np.ndim == 2:
        references_np = references_np[np.newaxis, ...]
        estimates_np = estimates_np[np.newaxis, ...]
        
    batch_size, n_sources, n_samples = references_np.shape
    
    # Convert frame and hop lengths to samples
    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(hop_length * sample_rate)
    
    # Calculate number of frames
    n_frames = 1 + (n_samples - frame_length_samples) // hop_length_samples
    
    # Initialize result dictionary
    results = {metric: np.zeros((batch_size, n_sources, n_frames)) for metric in metrics}
    time_stamps = np.arange(n_frames) * hop_length

    # Compute metrics for each frame
    for i in range(n_frames):
        start = i * hop_length_samples
        end = start + frame_length_samples
        
        if end > n_samples:
            end = n_samples
            
        frame_references = references_np[:, :, start:end]
        frame_estimates = estimates_np[:, :, start:end]
        
        # Skip frames that are too short
        if end - start < filter_length * 2:
            for metric in metrics:
                results[metric][:, :, i] = np.nan
            continue
            
        # Compute metrics for this frame
        frame_metrics = compute_metrics(
            frame_references,
            frame_estimates,
            sample_rate,
            metrics=metrics,
            compute_permutation=False,  # Use original source order for consistency
            filter_length=min(filter_length, (end - start) // 4),  # Adjust filter length for short frames
            zero_mean=zero_mean,
            use_cg_iter=use_cg_iter,
            load_diag=load_diag,
        )
        
        # Store results
        for metric in metrics:
            results[metric][:, :, i] = frame_metrics[metric]
    
    # Add time stamps to results
    results["time"] = time_stamps
    
    # Remove batch dimension if there was none originally
    if batch_size == 1:
        for metric in results:
            if metric == "time":
                continue
            results[metric] = results[metric][0]
    
    return results 