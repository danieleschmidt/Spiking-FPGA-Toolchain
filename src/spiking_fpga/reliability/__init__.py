"""
Advanced reliability and fault tolerance modules for production-ready FPGA deployment.

This module implements industry-grade reliability features including:
- Automatic error recovery and graceful degradation
- Real-time fault detection and isolation
- Redundant system configurations
- Advanced circuit breaker patterns
- Self-healing mechanisms
"""

from .fault_tolerance import FaultTolerantCompiler, RedundancyManager, FailureRecovery, CheckpointManager, FaultToleranceConfig, RedundancyMode
from .error_recovery import ErrorRecoverySystem, GracefulDegradation, CircuitBreakerAdvanced, ErrorContext

__all__ = [
    "FaultTolerantCompiler",
    "RedundancyManager", 
    "FailureRecovery",
    "CheckpointManager",
    "FaultToleranceConfig",
    "RedundancyMode",
    "ErrorRecoverySystem",
    "GracefulDegradation",
    "CircuitBreakerAdvanced",
    "ErrorContext",
]