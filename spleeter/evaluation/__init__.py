#!/usr/bin/env python
# coding: utf8

"""Evaluation module for Spleeter."""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

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
from spleeter.evaluation.evaluate import (
    separate_audio,
    load_reference_sources,
    evaluate_model
) 