"""
HierGWAS: Hierarchical Multi-Scale Attention for Genome-Wide Association Studies

This package implements HierGWAS, a novel deep learning architecture that captures
genomic interactions at multiple biological scales using hierarchical attention mechanisms.

Key Components:
- HierGWAS: Main model class
- HierarchicalAttention: Core attention mechanism
- GWASData: Data loading and preprocessing
- Utils: Utility functions for analysis and visualization

Authors: [Your Name]
License: MIT
Version: 1.0.0
"""

from .hiergwas import HierGWAS
from .attention import HierarchicalMultiScaleAttention
from .data import GWASData
from .utils import visualize_attention, compute_metrics
from .config import HierGWASConfig

__version__ = "1.0.0"
__author__ = "HierGWAS Team"
__email__ = "hiergwas@research.edu"

__all__ = [
    "HierGWAS",
    "HierarchicalMultiScaleAttention", 
    "GWASData",
    "HierGWASConfig",
    "visualize_attention",
    "compute_metrics"
]

