"""
Advanced performance optimization and scalability modules.

This module implements cutting-edge performance features including:
- Distributed multi-FPGA compilation
- Adaptive resource allocation
- Real-time performance monitoring
- ML-based optimization strategies
- Auto-scaling capabilities
- Advanced caching and memory optimization
"""

from .distributed_compiler import DistributedCompiler, ClusterManager, WorkloadBalancer
from .auto_scaling import AutoScaler, ResourcePredictor, ScalingPolicy
from .performance_optimizer import PerformanceOrchestrator, MLOptimizer, AdaptiveResourceAllocator
from .caching_advanced import IntelligentCache, DistributedCache, PredictiveCache

__all__ = [
    "DistributedCompiler",
    "ClusterManager",
    "WorkloadBalancer",
    "AutoScaler",
    "ResourcePredictor", 
    "ScalingPolicy",
    "PerformanceOrchestrator",
    "MLOptimizer",
    "AdaptiveResourceAllocator",
    "IntelligentCache",
    "DistributedCache",
    "PredictiveCache",
]