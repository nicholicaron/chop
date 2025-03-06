"""Benchmarking module for CHOP.

This module provides tools for creating, managing, and running benchmark suites
for various optimization problems.
"""

from .metrics import InstanceMetrics, SolverMetrics
from .suite import BenchmarkSuite, BenchmarkResult, save_results, load_results
from .runner import BenchmarkRunner

__all__ = [
    'InstanceMetrics',
    'SolverMetrics',
    'BenchmarkSuite',
    'BenchmarkResult',
    'BenchmarkRunner',
    'save_results',
    'load_results',
]