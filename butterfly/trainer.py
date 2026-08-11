"""Backward-compatible general training runtime imports."""
from .training.runtime import best_device, configure_cpu, continue_training

__all__ = ["best_device", "configure_cpu", "continue_training"]
