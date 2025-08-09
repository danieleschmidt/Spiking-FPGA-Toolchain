"""
Distributed Multi-FPGA Processing Module

Advanced distributed neuromorphic computing system supporting:
- Multi-FPGA orchestration and resource management
- Intelligent workload partitioning and load balancing
- Fault tolerance with automatic failover
- Inter-FPGA communication and synchronization
"""

from .multi_fpga_orchestrator import (
    MultiFPGAOrchestrator,
    FPGADevice,
    DistributedTask,
    PartitionedNetwork,
    FPGAState,
    TaskPriority,
    LoadBalancer,
    NetworkPartitioner,
    FaultToleranceManager,
    InterFPGACommunication,
    create_orchestrator,
    create_fpga_device,
    create_distributed_task
)

__all__ = [
    'MultiFPGAOrchestrator',
    'FPGADevice', 
    'DistributedTask',
    'PartitionedNetwork',
    'FPGAState',
    'TaskPriority',
    'LoadBalancer',
    'NetworkPartitioner', 
    'FaultToleranceManager',
    'InterFPGACommunication',
    'create_orchestrator',
    'create_fpga_device',
    'create_distributed_task'
]