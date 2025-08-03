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
from spiking_fpga.network_compiler import NetworkCompiler, compile_network

__all__ = [
    "FPGATarget",
    "compile_network", 
    "NetworkCompiler",
    "Network",
    "HDLGenerator",
    "VivadoBackend",
    "QuartusBackend",
]