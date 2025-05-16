#!/usr/bin/env python
# coding: utf8

"""Metrics module for evaluating audio source separation."""

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

from spleeter.evaluation.metrics.objective import (
    compute_metrics,
    frequency_dependent_sdr,
    compute_metrics_over_frames
) 