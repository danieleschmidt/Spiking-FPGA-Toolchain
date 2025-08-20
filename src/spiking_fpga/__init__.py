"""
Spiking-FPGA-Toolchain: Open-source toolchain for compiling spiking neural networks to FPGA hardware.

This package provides a comprehensive end-to-end pipeline for deploying neuromorphic 
computing models on commodity FPGAs, bridging the gap between high-level SNN frameworks 
and low-level hardware implementations.
"""

__version__ = "0.1.0-dev"
__author__ = "Spiking-FPGA-Toolchain Contributors"
__license__ = "Apache-2.0"

from spiking_fpga.core import FPGATarget
from spiking_fpga.compiler import HDLGenerator, VivadoBackend, QuartusBackend
from spiking_fpga.models import Network
from spiking_fpga.models.optimization import OptimizationLevel
from spiking_fpga.network_compiler import NetworkCompiler, compile_network

# Research modules for cutting-edge neuromorphic algorithms (lazy import to avoid heavy dependencies)
def _import_research_modules():
    try:
        from spiking_fpga.research import (
            AdaptiveSpikeCoder,
            MultiModalEncoder,
            MetaPlasticSTDP,
            BitstiftSTDP,
            HomeostasticRegulator,
        )
        return {
            'AdaptiveSpikeCoder': AdaptiveSpikeCoder,
            'MultiModalEncoder': MultiModalEncoder,
            'MetaPlasticSTDP': MetaPlasticSTDP,
            'BitstiftSTDP': BitstiftSTDP,
            'HomeostasticRegulator': HomeostasticRegulator,
        }
    except ImportError as e:
        import warnings
        warnings.warn(f"Research modules require additional dependencies: {e}")
        return {}

_research_modules = _import_research_modules()

# Advanced performance optimization
from spiking_fpga.performance_optimizer import (
    create_optimized_compiler,
    AdaptivePerformanceController,
    SystemResourceMonitor,
)

__all__ = [
    "FPGATarget",
    "compile_network", 
    "NetworkCompiler",
    "Network",
    "OptimizationLevel",
    "HDLGenerator",
    "VivadoBackend",
    "QuartusBackend",
    # Performance optimization
    "create_optimized_compiler",
    "AdaptivePerformanceController",
    "SystemResourceMonitor",
] + list(_research_modules.keys())

# Add research modules to globals if available
globals().update(_research_modules)