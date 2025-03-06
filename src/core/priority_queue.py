"""
Priority queue implementation for branch-and-bound.

This module provides a custom priority queue implementation that can
use different prioritization strategies and maintains a history of
operations for analysis.
"""

import heapq
from typing import Any, List, Tuple, Dict, Optional, Callable
import numpy as np


class PriorityQueue:
    """
    Priority queue implementation that supports different prioritization strategies.
    
    For branch-and-bound, this queue typically orders nodes by their bound values,
    but can be configured to use different strategies (best-first, depth-first, etc.).
    """
    
    def __init__(self, prioritizer=None):
        """
        Initialize a new priority queue.
        
        Args:
            prioritizer: Strategy object that defines node ordering
        """
        self.queue = []  # Underlying heap
        self.prioritizer = prioritizer
        
        # For error checking and debugging
        self.pushes = 0
        self.pops = 0
        self.item_count = 0
        
        # For analytics and visualization
        self.history = {
            "pushes": [],
            "pops": [],
            "lengths": []
        }
        
    def push(self, item: Any):
        """
        Add an item to the priority queue.
        
        Args:
            item: Item to add (must be comparable or have key defined in prioritizer)
        """
        if self.prioritizer:
            # Use the prioritizer to get a key for this item
            key = self.prioritizer.get_priority_key(item)
        else:
            # Default comparison using the item's value attribute
            key = (item.value,)
            
        # For a tuple key, we need to negate each component for max-heap behavior
        # Python's heapq implements a min-heap, so we negate values for max-heap
        if isinstance(key, tuple):
            # Negate each component for max-heap with tuple keys
            negated_key = tuple(-k for k in key)
        else:
            # Simple case for a single value
            negated_key = -key
            
        # Push to heap with a counter to ensure stable sorting for equal priorities
        heapq.heappush(self.queue, (negated_key, self.pushes, item))
        
        # Update counters
        self.pushes += 1
        self.item_count += 1
        
        # Record history
        self.history["pushes"].append({
            "item_id": getattr(item, "id", None),
            "priority": key,
            "queue_size": self.item_count
        })
        self.history["lengths"].append(self.item_count)
        
    def pop(self) -> Any:
        """
        Remove and return the highest priority item.
        
        Returns:
            The highest priority item
            
        Raises:
            IndexError: If the queue is empty
        """
        if not self.queue:
            raise IndexError("Pop from an empty priority queue")
            
        # Pop from heap
        _, _, item = heapq.heappop(self.queue)
        
        # Update counters
        self.pops += 1
        self.item_count -= 1
        
        # Record history
        self.history["pops"].append({
            "item_id": getattr(item, "id", None),
            "queue_size": self.item_count
        })
        self.history["lengths"].append(self.item_count)
        
        return item
        
    def peek(self) -> Any:
        """
        Return the highest priority item without removing it.
        
        Returns:
            The highest priority item
            
        Raises:
            IndexError: If the queue is empty
        """
        if not self.queue:
            raise IndexError("Peek from an empty priority queue")
            
        # Return item without popping
        return self.queue[0][2]
        
    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return self.item_count
        
    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self.queue)
        
    def items(self) -> List[Any]:
        """
        Return all items in the queue (in no particular order).
        
        Returns:
            List of all items
        """
        return [item for _, _, item in self.queue]
        
    def clear(self):
        """Clear the queue."""
        self.queue = []
        self.item_count = 0
        
    def set_prioritizer(self, prioritizer):
        """
        Change the prioritization strategy.
        
        Warning: This requires rebuilding the entire queue, which is O(n).
        
        Args:
            prioritizer: New prioritizer to use
        """
        if not self.queue:
            self.prioritizer = prioritizer
            return
            
        # Save all items
        items = self.items()
        
        # Clear and rebuild queue with new prioritizer
        self.clear()
        self.prioritizer = prioritizer
        
        for item in items:
            self.push(item)
            
    def get_statistics(self) -> Dict:
        """
        Get statistics about queue operations.
        
        Returns:
            Dict with queue statistics
        """
        stats = {
            "total_pushes": self.pushes,
            "total_pops": self.pops,
            "current_size": self.item_count,
            "max_size": max(self.history["lengths"]) if self.history["lengths"] else 0,
            "avg_size": np.mean(self.history["lengths"]) if self.history["lengths"] else 0
        }
        return stats