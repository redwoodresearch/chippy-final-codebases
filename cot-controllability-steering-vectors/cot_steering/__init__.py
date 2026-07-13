"""CoT-controllability steering vectors -- minimal release package.

Public surface:
  * ``cot_steering.figures`` -- regenerate the release figures from released summary artifacts.
  * ``cot_steering.artifacts`` -- locate artifacts (Hugging Face, local fallback).
  * ``cot_steering.instructions`` -- the 25-instruction CoT-control suite + scorers + splits.
  * ``cot_steering.scoring`` -- answer extraction + accuracy scoring (reference; not on the figure path).
  * ``cot_steering.steering`` -- load a steering ``.npz`` and the residual-stream apply hook.
"""
from . import artifacts, figures, instructions, scoring, steering  # noqa: F401

__all__ = ["figures", "artifacts", "instructions", "scoring", "steering"]
__version__ = "1.0.0"
