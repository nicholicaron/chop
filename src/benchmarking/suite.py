"""Benchmark suite management for CHOP."""

import os
import json
import time
import datetime
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import numpy as np

from ..problems.base import OptimizationProblem
from .metrics import InstanceMetrics, SolverMetrics


@dataclass
class BenchmarkResult:
    """Result of running a benchmark on a problem instance."""
    # Instance identification
    problem_type: str
    problem_name: str
    instance_id: str
    
    # Metrics
    instance_metrics: Dict[str, Any]  # Serialized InstanceMetrics
    solver_metrics: Dict[str, Any]  # Serialized SolverMetrics
    
    # Configuration
    solver_config: Dict[str, Any]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    duration: float = 0.0
    
    @classmethod
    def create(
        cls, 
        problem: OptimizationProblem, 
        instance_metrics: InstanceMetrics,
        solver_metrics: SolverMetrics, 
        solver_config: Dict[str, Any],
        duration: float
    ) -> 'BenchmarkResult':
        """Create a benchmark result from metrics and configuration."""
        return cls(
            problem_type=problem.__class__.__name__,
            problem_name=problem.name,
            instance_id=f"{problem.__class__.__name__}_{problem.name}",
            instance_metrics=asdict(instance_metrics),
            solver_metrics=asdict(solver_metrics),
            solver_config=solver_config,
            duration=duration
        )


class BenchmarkSuite:
    """A collection of problem instances for benchmarking."""
    
    def __init__(self, name: str):
        """Initialize a benchmark suite.
        
        Args:
            name: Name of the benchmark suite
        """
        self.name = name
        self.instances: Dict[str, List[OptimizationProblem]] = {}
        self.metrics: Dict[str, Dict[str, InstanceMetrics]] = {}
    
    def add_instances(self, problem_type: str, instances: List[OptimizationProblem]) -> None:
        """Add problem instances to the benchmark suite.
        
        Args:
            problem_type: Type of problem (e.g., 'TSP', 'Knapsack')
            instances: List of problem instances
        """
        if problem_type not in self.instances:
            self.instances[problem_type] = []
            self.metrics[problem_type] = {}
        
        self.instances[problem_type].extend(instances)
        
        # Calculate metrics for each instance
        for instance in instances:
            instance_id = f"{problem_type}_{instance.name}"
            self.metrics[problem_type][instance_id] = InstanceMetrics.calculate(instance)
    
    def get_instance(self, problem_type: str, instance_name: str) -> Optional[OptimizationProblem]:
        """Get a problem instance by type and name.
        
        Args:
            problem_type: Type of problem (e.g., 'TSP', 'Knapsack')
            instance_name: Name of the instance
            
        Returns:
            The problem instance if found, None otherwise
        """
        if problem_type not in self.instances:
            return None
        
        for instance in self.instances[problem_type]:
            if instance.name == instance_name:
                return instance
        
        return None
    
    def get_metrics(self, problem_type: str, instance_name: str) -> Optional[InstanceMetrics]:
        """Get metrics for a problem instance.
        
        Args:
            problem_type: Type of problem (e.g., 'TSP', 'Knapsack')
            instance_name: Name of the instance
            
        Returns:
            The instance metrics if found, None otherwise
        """
        if problem_type not in self.metrics:
            return None
        
        instance_id = f"{problem_type}_{instance_name}"
        return self.metrics[problem_type].get(instance_id)
    
    def get_instances_by_difficulty(self, problem_type: str, difficulty: str) -> List[OptimizationProblem]:
        """Get all instances of a given problem type and difficulty.
        
        Args:
            problem_type: Type of problem (e.g., 'TSP', 'Knapsack')
            difficulty: Difficulty level (e.g., 'easy', 'medium', 'hard')
            
        Returns:
            List of problem instances
        """
        if problem_type not in self.instances:
            return []
        
        return [
            instance for instance in self.instances[problem_type]
            if instance.difficulty == difficulty
        ]
    
    def get_all_instances(self) -> Iterator[Tuple[str, OptimizationProblem]]:
        """Get all instances in the benchmark suite.
        
        Returns:
            Iterator of (problem_type, instance) pairs
        """
        for problem_type, instances in self.instances.items():
            for instance in instances:
                yield problem_type, instance
    
    def get_problem_types(self) -> List[str]:
        """Get all problem types in the benchmark suite.
        
        Returns:
            List of problem type names
        """
        return list(self.instances.keys())
    
    def count_instances(self, problem_type: Optional[str] = None) -> int:
        """Count the number of instances in the benchmark suite.
        
        Args:
            problem_type: If provided, count only instances of this type
            
        Returns:
            Number of instances
        """
        if problem_type is not None:
            return len(self.instances.get(problem_type, []))
        else:
            return sum(len(instances) for instances in self.instances.values())
    
    @classmethod
    def from_problem_generators(
        cls, 
        name: str, 
        problem_classes: List[type], 
        difficulties: List[str] = None
    ) -> 'BenchmarkSuite':
        """Create a benchmark suite from problem generators.
        
        Args:
            name: Name of the benchmark suite
            problem_classes: List of problem classes (e.g., [TSP, Knapsack])
            difficulties: List of difficulty levels to include (default: ['easy', 'medium', 'hard'])
            
        Returns:
            A benchmark suite containing instances for each problem type and difficulty
        """
        suite = cls(name)
        
        if difficulties is None:
            difficulties = ['easy', 'medium', 'hard']
        
        for problem_class in problem_classes:
            # Generate benchmark instances
            instances_by_difficulty = problem_class.generate_benchmark_suite(difficulties)
            
            # Flatten instances
            all_instances = []
            for difficulty in difficulties:
                if difficulty in instances_by_difficulty:
                    all_instances.extend(instances_by_difficulty[difficulty])
            
            # Add instances to the suite
            suite.add_instances(problem_class.__name__, all_instances)
        
        return suite
    
    @classmethod
    def from_predefined_instances(cls, name: str, problem_classes: List[type]) -> 'BenchmarkSuite':
        """Create a benchmark suite from predefined instances.
        
        Args:
            name: Name of the benchmark suite
            problem_classes: List of problem classes (e.g., [TSP, Knapsack])
            
        Returns:
            A benchmark suite containing predefined instances for each problem type
        """
        suite = cls(name)
        
        for problem_class in problem_classes:
            # Get predefined instances
            predefined_instances = problem_class.create_predefined_instances()
            
            # Add instances to the suite
            suite.add_instances(problem_class.__name__, list(predefined_instances.values()))
        
        return suite


def save_results(results: List[BenchmarkResult], output_dir: str, filename: Optional[str] = None) -> str:
    """Save benchmark results to disk.
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save results to
        filename: Name of the file to save results to (default: benchmark_results_{timestamp}.json)
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results_{timestamp}.json"
    
    # Convert results to dictionaries
    results_dict = [asdict(result) for result in results]
    
    # Save results to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return output_path


def load_results(filepath: str) -> List[BenchmarkResult]:
    """Load benchmark results from disk.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        List of benchmark results
    """
    with open(filepath, 'r') as f:
        results_dict = json.load(f)
    
    # Convert dictionaries to BenchmarkResult objects
    results = [BenchmarkResult(**result) for result in results_dict]
    
    return results