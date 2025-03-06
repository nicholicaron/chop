"""
Benchmark visualization script for CHOP.

This script loads and visualizes benchmark results that were previously saved to disk.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarking import (
    BenchmarkRunner, 
    BenchmarkSuite, 
    BenchmarkResult,
    load_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the benchmark results JSON file"
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for visualizations (default: results_<timestamp>)"
    )
    
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for visualization files (default: png)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare solver configurations"
    )
    
    return parser.parse_args()


def main():
    """Main function to visualize benchmark results."""
    # Parse command line arguments
    args = parse_args()
    
    # Load results
    results_path = Path(args.results)
    print(f"Loading benchmark results from {results_path}")
    results = load_results(results_path)
    
    if not results:
        print("No results found in the specified file")
        return
    
    print(f"Loaded {len(results)} benchmark results")
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = results_path.stem.split('_')[-1]
        output_dir = f"results_viz_{timestamp}"
    
    # Create dummy suite and runner for visualization
    suite = BenchmarkSuite("dummy")
    runner = BenchmarkRunner(suite, output_dir=output_dir)
    
    # Visualize results
    print(f"Generating visualizations in {output_dir}")
    runner.visualize_results(results, save_format=args.format)
    
    # Compare solver configurations if requested
    if args.compare:
        print("Comparing solver configurations")
        runner.compare_solvers(results, save_format=args.format)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()