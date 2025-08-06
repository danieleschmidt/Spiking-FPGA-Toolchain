"""
Research module for novel neuromorphic algorithms and experimental features.

This module contains cutting-edge research implementations including:
- Bio-inspired adaptive spike encoding/decoding
- Hardware-optimized STDP with meta-plasticity  
- Neuromorphic graph neural networks
- Quantum-inspired optimization algorithms
- Federated learning with differential privacy
"""

from .adaptive_encoding import AdaptiveSpikeCoder, MultiModalEncoder
from .meta_plasticity import MetaPlasticSTDP, BitstiftSTDP, HomeostasticRegulator

__all__ = [
    "AdaptiveSpikeCoder",
    "MultiModalEncoder", 
    "MetaPlasticSTDP",
    "BitstiftSTDP",
    "HomeostasticRegulator",
]