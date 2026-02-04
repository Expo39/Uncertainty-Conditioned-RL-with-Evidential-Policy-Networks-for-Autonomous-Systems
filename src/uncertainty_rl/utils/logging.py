"""Logging utilities for uncertainty-conditioned RL.

This module provides logging utilities for tracking training progress and
uncertainty metrics.
"""
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import json
import csv


class MetricsLogger:
    """Logger for tracking training and evaluation metrics.
    
    Args:
        log_dir: Directory to save logs.
        prefix: Prefix for log files.
    """
    
    def __init__(self, log_dir: str, prefix: str = "metrics") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        
        self.metrics: Dict[str, List[Any]] = {}
        
    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a given step.
        
        Args:
            step: Training step or episode number.
            metrics: Dictionary of metric names to values.
        """
        if "step" not in self.metrics:
            self.metrics["step"] = []
        self.metrics["step"].append(step)
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def save_csv(self, filename: Optional[str] = None) -> None:
        """Save metrics to CSV file.
        
        Args:
            filename: Optional custom filename.
        """
        if filename is None:
            filename = f"{self.prefix}.csv"
            
        filepath = self.log_dir / filename
        
        if not self.metrics:
            return
            
        # Get all keys
        keys = list(self.metrics.keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            
            # Write rows
            num_rows = len(self.metrics[keys[0]])
            for i in range(num_rows):
                row = {key: self.metrics[key][i] for key in keys}
                writer.writerow(row)
                
    def save_json(self, filename: Optional[str] = None) -> None:
        """Save metrics to JSON file.
        
        Args:
            filename: Optional custom filename.
        """
        if filename is None:
            filename = f"{self.prefix}.json"
            
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_metric(self, name: str) -> Optional[List[Any]]:
        """Get logged values for a specific metric.
        
        Args:
            name: Metric name.
            
        Returns:
            List of logged values or None if metric doesn't exist.
        """
        return self.metrics.get(name)
    
    def compute_statistics(self, name: str) -> Dict[str, float]:
        """Compute statistics for a metric.
        
        Args:
            name: Metric name.
            
        Returns:
            Dictionary with mean, std, min, max.
        """
        values = self.get_metric(name)
        if values is None:
            return {}
            
        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }


class UncertaintyTracker:
    """Tracker for epistemic and aleatoric uncertainty during training.
    
    Args:
        window_size: Size of sliding window for computing statistics.
    """
    
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.epistemic_values: List[float] = []
        self.aleatoric_values: List[float] = []
        
    def update(self, epistemic: float, aleatoric: float) -> None:
        """Update tracker with new uncertainty values.
        
        Args:
            epistemic: Epistemic uncertainty value.
            aleatoric: Aleatoric uncertainty value.
        """
        self.epistemic_values.append(epistemic)
        self.aleatoric_values.append(aleatoric)
        
        # Keep only recent values
        if len(self.epistemic_values) > self.window_size:
            self.epistemic_values.pop(0)
            self.aleatoric_values.pop(0)
            
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for tracked uncertainties.
        
        Returns:
            Dictionary with statistics for epistemic and aleatoric uncertainty.
        """
        if not self.epistemic_values:
            return {}
            
        epistemic_array = np.array(self.epistemic_values)
        aleatoric_array = np.array(self.aleatoric_values)
        
        return {
            "epistemic": {
                "mean": float(np.mean(epistemic_array)),
                "std": float(np.std(epistemic_array)),
                "min": float(np.min(epistemic_array)),
                "max": float(np.max(epistemic_array)),
            },
            "aleatoric": {
                "mean": float(np.mean(aleatoric_array)),
                "std": float(np.std(aleatoric_array)),
                "min": float(np.min(aleatoric_array)),
                "max": float(np.max(aleatoric_array)),
            },
        }
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.epistemic_values.clear()
        self.aleatoric_values.clear()
