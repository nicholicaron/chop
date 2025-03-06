"""Metrics for characterizing problem instances and solver performance."""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union

from ..core.solver import BnBSolver
from ..problems.base import OptimizationProblem


@dataclass
class InstanceMetrics:
    """Metrics that characterize a problem instance.
    
    These metrics are calculated before solving and capture structural
    properties of the instance that may correlate with difficulty.
    """
    # Basic problem properties
    problem_type: str
    problem_name: str
    difficulty: str
    size: Dict[str, int]  # Problem-specific size metrics (e.g., num_cities for TSP)
    
    # LP relaxation properties
    lp_relaxation_value: float
    lp_relaxation_time: float  # Time to solve LP relaxation
    
    # Theoretical complexity measures
    num_variables: int
    num_constraints: int
    density: float  # Percentage of non-zero coefficients in the constraint matrix
    
    # Problem-specific metrics
    specific_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def lp_relaxation_gap(self) -> Optional[float]:
        """Calculate gap between LP relaxation and optimal (if known)."""
        if hasattr(self, 'optimal_value') and self.optimal_value is not None:
            # Handle minimization and maximization problems
            if self.specific_metrics.get('is_minimization', True):
                # For minimization: (optimal - relaxation) / optimal
                if self.optimal_value == 0:
                    return 0 if self.lp_relaxation_value == 0 else float('inf')
                return (self.optimal_value - self.lp_relaxation_value) / abs(self.optimal_value)
            else:
                # For maximization: (relaxation - optimal) / relaxation
                if self.lp_relaxation_value == 0:
                    return 0 if self.optimal_value == 0 else float('inf')
                return (self.lp_relaxation_value - self.optimal_value) / abs(self.lp_relaxation_value)
        return None
    
    @classmethod
    def calculate(cls, problem: OptimizationProblem) -> 'InstanceMetrics':
        """Calculate metrics for a problem instance."""
        # Get basic problem information
        problem_type = problem.__class__.__name__
        problem_name = problem.name
        difficulty = problem.difficulty
        
        # Get problem-specific size metrics
        size = problem.size_metrics()
        
        # Solve LP relaxation
        start_time = time.time()
        lp_relaxation_value = problem.solve_lp_relaxation()
        lp_relaxation_time = time.time() - start_time
        
        # Get problem structure information
        ilp_model = problem.to_ilp()
        num_variables = len(ilp_model['variable_types'])
        num_constraints = len(ilp_model['constraints'])
        
        # Calculate constraint matrix density
        total_cells = num_variables * num_constraints
        non_zero_cells = sum(len(constraint.get('variables', [])) 
                           for constraint in ilp_model['constraints'])
        density = non_zero_cells / total_cells if total_cells > 0 else 0
        
        # Get problem-specific metrics
        specific_metrics = problem.get_specific_metrics()
        
        return cls(
            problem_type=problem_type,
            problem_name=problem_name,
            difficulty=difficulty,
            size=size,
            lp_relaxation_value=lp_relaxation_value,
            lp_relaxation_time=lp_relaxation_time,
            num_variables=num_variables,
            num_constraints=num_constraints,
            density=density,
            specific_metrics=specific_metrics
        )


@dataclass
class SolverMetrics:
    """Metrics for evaluating solver performance on a problem instance."""
    # Basic solution information
    solution_found: bool
    solution_value: Optional[float] = None
    solution_time: float = 0.0
    proven_optimal: bool = False
    
    # Branch-and-bound statistics
    nodes_created: int = 0
    nodes_processed: int = 0
    nodes_pruned_bound: int = 0
    nodes_pruned_infeasible: int = 0
    nodes_integer_feasible: int = 0
    
    # Computational efficiency
    lp_relaxations_solved: int = 0
    lp_solving_time: float = 0.0
    branching_time: float = 0.0
    node_processing_time: float = 0.0
    
    # Solution quality
    optimality_gap: Optional[float] = None
    best_bound: Optional[float] = None
    
    # Additional solver statistics
    solver_specific: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_solver(cls, solver: BnBSolver) -> 'SolverMetrics':
        """Extract metrics from a solver after solving a problem."""
        stats = solver.get_statistics()
        
        # Basic solution information
        solution_found = solver.best_int_feasible_node is not None
        solution_value = solver.optimal_obj_value if solution_found else None
        proven_optimal = solver.status == 'optimal'
        
        # Calculate optimality gap
        best_bound = solver.global_lower_bound
        optimality_gap = None
        if solution_found and best_bound is not None:
            # For minimization problems
            if abs(solution_value) < 1e-10:
                optimality_gap = 0.0 if abs(best_bound) < 1e-10 else float('inf')
            else:
                optimality_gap = (solution_value - best_bound) / abs(solution_value)
        
        return cls(
            solution_found=solution_found,
            solution_value=solution_value,
            solution_time=stats['times']['total'],
            proven_optimal=proven_optimal,
            nodes_created=stats['nodes']['created'],
            nodes_processed=stats['nodes']['processed'],
            nodes_pruned_bound=stats['nodes']['pruned_bound'],
            nodes_pruned_infeasible=stats['nodes']['pruned_infeasible'],
            nodes_integer_feasible=stats['nodes']['integer_feasible'],
            lp_relaxations_solved=stats['lp_relaxations'],
            lp_solving_time=stats['times']['lp_solving'],
            branching_time=stats['times']['branching'],
            node_processing_time=stats['times']['node_processing'],
            optimality_gap=optimality_gap,
            best_bound=best_bound,
            solver_specific={
                'num_gomory_cuts': stats.get('cuts_added', 0),
                'early_stopped': stats.get('early_stopped', False),
                'status': solver.status
            }
        )