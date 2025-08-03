"""
Core services for SNN compilation and FPGA deployment.
"""

from .network_compiler import NetworkCompiler, CompilationResult
from .hdl_generator import HDLGenerator, VerilogTemplate
from .resource_mapper import ResourceMapper, PlacementResult
from .optimization_pipeline import OptimizationPipeline, OptimizationPass

__all__ = [
    'NetworkCompiler', 'CompilationResult',
    'HDLGenerator', 'VerilogTemplate', 
    'ResourceMapper', 'PlacementResult',
    'OptimizationPipeline', 'OptimizationPass'
]