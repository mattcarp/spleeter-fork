#!/usr/bin/env python
# coding: utf8

""" Unit tests for evaluation metrics. """

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

import numpy as np
import pytest
from tempfile import TemporaryDirectory
import os
import matplotlib.pyplot as plt

from spleeter.evaluation.metrics.objective import (
    compute_metrics,
    frequency_dependent_sdr,
    compute_metrics_over_frames
)
from spleeter.evaluation.visualization.plots import (
    plot_metrics_comparison,
    plot_metrics_over_time,
    plot_frequency_dependent_metrics
)


@pytest.fixture
def dummy_sources():
    """Create dummy reference and estimated audio sources for testing."""
    np.random.seed(42)  # For reproducibility
    sample_rate = 44100
    duration = 2.0  # seconds
    n_samples = int(sample_rate * duration)
    n_sources = 2
    
    # Create reference signals: two sine waves with different frequencies
    t = np.linspace(0, duration, n_samples)
    references = np.zeros((n_sources, n_samples))
    references[0] = 0.8 * np.sin(2 * np.pi * 440 * t)  # A4 note
    references[1] = 0.5 * np.sin(2 * np.pi * 880 * t)  # A5 note
    
    # Create estimated signals: same frequencies but with noise and slight time shift
    estimates = np.zeros((n_sources, n_samples))
    estimates[0] = 0.7 * np.sin(2 * np.pi * 440 * (t + 0.001)) + 0.1 * np.random.randn(n_samples)
    estimates[1] = 0.6 * np.sin(2 * np.pi * 880 * (t - 0.002)) + 0.15 * np.random.randn(n_samples)
    
    return references, estimates, sample_rate


def test_compute_metrics(dummy_sources):
    """Test basic metrics computation."""
    references, estimates, sample_rate = dummy_sources
    
    # Compute all available metrics
    metrics = compute_metrics(
        references, 
        estimates, 
        sample_rate,
        metrics=["sdr", "sir", "sar", "si-sdr"],
        compute_permutation=True
    )
    
    # Check that all metrics are computed
    assert "sdr" in metrics
    assert "sir" in metrics
    assert "sar" in metrics
    assert "si-sdr" in metrics
    assert "permutation" in metrics
    
    # Check shape of metrics
    assert metrics["sdr"].shape == (2,)  # One value per source
    assert metrics["sir"].shape == (2,)
    assert metrics["sar"].shape == (2,)
    assert metrics["si-sdr"].shape == (2,)
    
    # Check that metrics have reasonable values (SDR and SI-SDR should be positive for our test case)
    assert np.all(metrics["sdr"] > 0)
    assert np.all(metrics["si-sdr"] > 0)


def test_frequency_dependent_sdr(dummy_sources):
    """Test frequency-dependent SDR computation."""
    references, estimates, sample_rate = dummy_sources
    
    # Compute frequency-dependent SDR
    freq_sdr, band_freqs = frequency_dependent_sdr(
        references,
        estimates,
        sample_rate,
        n_bands=4
    )
    
    # Check shapes
    assert freq_sdr.shape == (2, 4)  # (n_sources, n_bands)
    assert len(band_freqs) == 4  # n_bands
    
    # Check frequency bands are ascending
    assert np.all(np.diff(band_freqs) > 0)
    
    # Check values are reasonable
    assert np.all(np.isfinite(freq_sdr))


def test_compute_metrics_over_frames(dummy_sources):
    """Test metrics computation over time frames."""
    references, estimates, sample_rate = dummy_sources
    
    # Compute metrics over frames
    frame_metrics = compute_metrics_over_frames(
        references,
        estimates,
        sample_rate,
        frame_length=0.5,  # Short frames for the test
        hop_length=0.25,
        metrics=["sdr", "sir"]
    )
    
    # Check that metrics and time are present
    assert "sdr" in frame_metrics
    assert "sir" in frame_metrics
    assert "time" in frame_metrics
    
    # Check shapes
    n_frames = len(frame_metrics["time"])
    assert frame_metrics["sdr"].shape == (2, n_frames)  # (n_sources, n_frames)
    assert frame_metrics["sir"].shape == (2, n_frames)
    
    # Check that metrics have reasonable values
    assert np.all(np.isfinite(frame_metrics["sdr"]))
    assert np.all(np.isfinite(frame_metrics["sir"]))


def test_visualization_plots(dummy_sources):
    """Test visualization functions."""
    references, estimates, sample_rate = dummy_sources
    
    # Compute metrics for visualization
    metrics = compute_metrics(
        references, 
        estimates, 
        sample_rate,
        metrics=["sdr", "sir", "sar", "si-sdr"]
    )
    
    frame_metrics = compute_metrics_over_frames(
        references,
        estimates,
        sample_rate,
        frame_length=0.5,
        hop_length=0.25,
        metrics=["sdr"]
    )
    
    freq_sdr, band_freqs = frequency_dependent_sdr(
        references,
        estimates,
        sample_rate,
        n_bands=4
    )
    
    # Create temporary directory for test output
    with TemporaryDirectory() as tmp_dir:
        # Test metrics comparison plot
        metrics_dict = {
            "model1": {"sdr": metrics["sdr"], "sir": metrics["sir"]},
            "model2": {"sdr": metrics["sdr"] * 0.9, "sir": metrics["sir"] * 1.1}  # Simulated second model
        }
        
        plt.switch_backend('agg')  # Non-interactive backend for testing
        
        # Test plot_metrics_comparison
        fig1 = plot_metrics_comparison(
            metrics_dict,
            metric_name="sdr",
            source_names=["vocals", "accompaniment"],
            save_path=os.path.join(tmp_dir, "comparison.png"),
            show_plot=False
        )
        assert isinstance(fig1, plt.Figure)
        
        # Test plot_metrics_over_time
        fig2 = plot_metrics_over_time(
            frame_metrics,
            source_names=["vocals", "accompaniment"],
            save_path=os.path.join(tmp_dir, "over_time.png"),
            show_plot=False
        )
        assert isinstance(fig2, plt.Figure)
        
        # Test plot_frequency_dependent_metrics
        fig3 = plot_frequency_dependent_metrics(
            freq_sdr,
            band_freqs,
            source_names=["vocals", "accompaniment"],
            save_path=os.path.join(tmp_dir, "freq_dependent.png"),
            show_plot=False
        )
        assert isinstance(fig3, plt.Figure)
        
        # Check that files were created
        assert os.path.exists(os.path.join(tmp_dir, "comparison.png"))
        assert os.path.exists(os.path.join(tmp_dir, "over_time.png"))
        assert os.path.exists(os.path.join(tmp_dir, "freq_dependent.png")) 