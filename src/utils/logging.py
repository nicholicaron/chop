"""
Logging and performance monitoring utilities.

This module provides tools for logging, timing, and performance profiling
of the branch-and-bound algorithm.
"""

import time
import sys
import os
import json
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from datetime import datetime


class Timer:
    """
    Simple timer class for measuring execution time.
    
    Can be used as a context manager or standalone.
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a timer.
        
        Args:
            name: Optional name for this timer
        """
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
        
    def __enter__(self):
        """Start timing when entering a context."""
        self.start()
        return self
        
    def __exit__(self, *args):
        """Stop timing when exiting a context."""
        self.stop()
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
        
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
        
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0
        
    def __str__(self):
        """Return formatted elapsed time."""
        return f"{self.name + ': ' if self.name else ''}{self.elapsed:.6f} seconds"


class BnBLogger:
    """
    Logger for branch-and-bound optimization.
    
    Tracks stats and events during the optimization process.
    """
    
    def __init__(self, log_file: str = None, verbose: bool = True):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to log file (or None for stdout only)
            verbose: Whether to print info to stdout
        """
        self.log_file = log_file
        self.verbose = verbose
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "nodes": {
                "created": 0,
                "processed": 0,
                "pruned_bound": 0,
                "pruned_infeasible": 0,
                "integer_feasible": 0,
                "optimal": 0
            },
            "lp_relaxations": 0,
            "cuts_added": 0,
            "best_bound": float("inf"),
            "best_objective": float("-inf"),
            "elapsed_time": 0.0,
            "events": []
        }
        
        self.timers = {
            "total": Timer("Total"),
            "lp_solving": Timer("LP Solving"),
            "branching": Timer("Branching"),
            "node_processing": Timer("Node Processing")
        }
        
        # Start the total timer
        self.timers["total"].start()
        
        # Set up file logging if requested
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.file = open(log_file, 'w')
        else:
            self.file = None
            
    def __del__(self):
        """Clean up resources when logger is destroyed."""
        if self.file:
            self.file.close()
            
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        
        if self.verbose:
            print(log_entry)
            
        if self.file:
            self.file.write(log_entry + "\n")
            self.file.flush()
            
        # Record significant events
        if level != "DEBUG":
            self.stats["events"].append({
                "time": timestamp,
                "level": level,
                "message": message
            })
            
    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, "DEBUG")
        
    def info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")
        
    def warning(self, message: str):
        """Log a warning message."""
        self.log(message, "WARNING")
        
    def error(self, message: str):
        """Log an error message."""
        self.log(message, "ERROR")
        
    def update_stats(self, key: str, value: Any):
        """
        Update a specific statistic.
        
        Args:
            key: Statistic to update
            value: New value
        """
        # Handle nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            current = self.stats
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.stats[key] = value
            
    def increment_stat(self, key: str, increment: int = 1):
        """
        Increment a counter statistic.
        
        Args:
            key: Statistic to increment
            increment: Amount to increment by
        """
        # Handle nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            current = self.stats
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            if parts[-1] not in current:
                current[parts[-1]] = 0
            current[parts[-1]] += increment
        else:
            if key not in self.stats:
                self.stats[key] = 0
            self.stats[key] += increment
    
    def start_timer(self, name: str):
        """
        Start a named timer.
        
        Args:
            name: Timer name
        """
        if name not in self.timers:
            self.timers[name] = Timer(name)
        self.timers[name].start()
        
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            float: Elapsed time in seconds
        """
        if name in self.timers:
            elapsed = self.timers[name].stop()
            if name != "total":  # Don't log the main timer until finish()
                self.debug(f"Timer {name}: {elapsed:.6f} seconds")
            return elapsed
        return 0.0
        
    def finish(self):
        """
        Finalize logging and save statistics.
        
        Stops the main timer and writes statistics to file.
        """
        # Stop the total timer
        total_time = self.stop_timer("total")
        self.stats["elapsed_time"] = total_time
        self.info(f"Total execution time: {total_time:.6f} seconds")
        
        # Log all timer results
        for name, timer in self.timers.items():
            if name != "total":  # Already logged
                self.info(f"Timer {name}: {timer.elapsed:.6f} seconds")
        
        # Save statistics to file
        if self.file:
            self.file.write("\n--- FINAL STATISTICS ---\n")
            self.file.write(json.dumps(self.stats, indent=2))
            self.file.write("\n")
            self.file.flush()
            
        return self.stats


class OutputRedirector:
    """
    Utility class for redirecting output to both console and log file.
    
    Enables simultaneous writing of output to both the terminal and a log file,
    useful for debugging and analysis of the branch-and-bound process.
    """
    
    def __init__(self, filename: str):
        """
        Initialize with log filename.
        
        Args:
            filename: Path to log file
        """
        self.terminal = sys.stdout
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log = open(filename, 'w')

    def write(self, message: str):
        """
        Write message to both terminal and log file.
        
        Args:
            message: Message to write
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush both output streams."""
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        """Close the log file."""
        self.log.close()


def time_function(logger):
    """
    Decorator for timing function execution.
    
    Args:
        logger: BnBLogger instance to use for timing
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = f"func.{func.__name__}"
            logger.start_timer(timer_name)
            result = func(*args, **kwargs)
            logger.stop_timer(timer_name)
            return result
        return wrapper
    return decorator