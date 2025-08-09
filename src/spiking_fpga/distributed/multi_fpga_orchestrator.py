"""
Multi-FPGA Distributed Processing Orchestrator

Advanced distributed neuromorphic computing system supporting:
- Intelligent workload partitioning across multiple FPGAs
- Real-time load balancing and resource optimization
- Fault tolerance with automatic failover and recovery
- Dynamic scaling and resource allocation
- Inter-FPGA communication and synchronization
"""

import time
import numpy as np
import threading
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import uuid
from enum import Enum
import socket
import struct

logger = logging.getLogger(__name__)


class FPGAState(Enum):
    """FPGA device states."""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    OVERLOADED = "overloaded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    REALTIME = 5


@dataclass
class FPGADevice:
    """FPGA device representation."""
    device_id: str
    hostname: str
    port: int
    fpga_type: str
    capabilities: Dict[str, Any]
    current_state: FPGAState = FPGAState.OFFLINE
    current_load: float = 0.0
    total_resources: Dict[str, int] = field(default_factory=dict)
    available_resources: Dict[str, int] = field(default_factory=dict)
    task_queue_size: int = 0
    last_heartbeat: float = 0.0
    error_count: int = 0
    total_tasks_completed: int = 0
    average_task_time: float = 0.0
    
    def __post_init__(self):
        if not self.total_resources:
            # Default resource estimates based on FPGA type
            if 'artix7' in self.fpga_type.lower():
                self.total_resources = {
                    'logic_cells': 33280 if '35t' in self.fpga_type else 101440,
                    'bram_kb': 1800 if '35t' in self.fpga_type else 4860,
                    'dsp_slices': 90 if '35t' in self.fpga_type else 240
                }
            elif 'cyclone' in self.fpga_type.lower():
                self.total_resources = {
                    'logic_elements': 77000,
                    'memory_blocks': 294,
                    'dsp_blocks': 150
                }
            else:
                self.total_resources = {'generic_units': 1000}
                
        if not self.available_resources:
            self.available_resources = self.total_resources.copy()


@dataclass
class DistributedTask:
    """Distributed neuromorphic computing task."""
    task_id: str
    task_type: str
    priority: TaskPriority
    network_config: Dict[str, Any]
    input_data: Optional[np.ndarray] = None
    expected_output_size: int = 0
    max_execution_time: float = 300.0  # 5 minutes default
    resource_requirements: Dict[str, int] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    callback_url: Optional[str] = None
    created_time: float = field(default_factory=time.time)
    assigned_device: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class PartitionedNetwork:
    """Network partition for distributed processing."""
    partition_id: str
    assigned_device: str
    neuron_indices: List[int]
    synapse_connections: List[Tuple[int, int, float]]
    input_mappings: Dict[int, int]
    output_mappings: Dict[int, int]
    inter_partition_connections: List[Tuple[str, int, int]] = field(default_factory=list)


class LoadBalancer:
    """Intelligent load balancer for FPGA resources."""
    
    def __init__(self):
        self.device_loads = {}
        self.load_history = defaultdict(lambda: deque(maxlen=100))
        self.predictor = None  # Could integrate ML predictor
        
    def calculate_device_score(self, device: FPGADevice, task: DistributedTask) -> float:
        """Calculate suitability score for assigning task to device."""
        if device.current_state not in [FPGAState.READY, FPGAState.PROCESSING]:
            return 0.0
            
        score = 100.0
        
        # Resource availability score
        resource_score = self._calculate_resource_score(device, task)
        score *= resource_score
        
        # Current load penalty
        load_penalty = 1.0 - (device.current_load / 100.0)
        score *= load_penalty
        
        # Queue size penalty
        queue_penalty = 1.0 - min(device.task_queue_size / 50.0, 0.8)
        score *= queue_penalty
        
        # Performance history bonus
        if device.total_tasks_completed > 0:
            performance_bonus = 1.0 + (1.0 / (device.average_task_time + 1.0))
            score *= performance_bonus
            
        # Error rate penalty
        if device.total_tasks_completed > 0:
            error_rate = device.error_count / device.total_tasks_completed
            error_penalty = 1.0 - min(error_rate, 0.9)
            score *= error_penalty
            
        # Priority boost for critical tasks
        if task.priority == TaskPriority.CRITICAL:
            score *= 1.5
        elif task.priority == TaskPriority.REALTIME:
            score *= 2.0
            
        return max(0.0, score)
        
    def _calculate_resource_score(self, device: FPGADevice, task: DistributedTask) -> float:
        """Calculate resource availability score."""
        if not task.resource_requirements:
            return 1.0
            
        total_score = 0.0
        num_resources = 0
        
        for resource, required in task.resource_requirements.items():
            available = device.available_resources.get(resource, 0)
            if required > 0:
                if available >= required:
                    resource_ratio = available / required
                    # Prefer devices with more available resources
                    total_score += min(2.0, resource_ratio)
                else:
                    # Insufficient resources
                    return 0.0
                num_resources += 1
                
        return total_score / max(1, num_resources)
        
    def select_best_device(self, devices: List[FPGADevice], 
                          task: DistributedTask) -> Optional[FPGADevice]:
        """Select best device for task assignment."""
        if not devices:
            return None
            
        device_scores = []
        for device in devices:
            score = self.calculate_device_score(device, task)
            device_scores.append((device, score))
            
        # Sort by score (descending)
        device_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best device if score > 0
        if device_scores[0][1] > 0:
            return device_scores[0][0]
        else:
            return None
            
    def update_device_load(self, device_id: str, new_load: float) -> None:
        """Update device load information."""
        self.device_loads[device_id] = new_load
        self.load_history[device_id].append((time.time(), new_load))


class NetworkPartitioner:
    """Intelligent network partitioning for distributed processing."""
    
    def __init__(self):
        self.partitioning_strategies = {
            'balanced': self._balanced_partitioning,
            'minimum_cut': self._minimum_cut_partitioning,
            'resource_aware': self._resource_aware_partitioning,
            'latency_optimized': self._latency_optimized_partitioning
        }
        
    def partition_network(self, 
                         network_config: Dict[str, Any],
                         available_devices: List[FPGADevice],
                         strategy: str = 'resource_aware') -> List[PartitionedNetwork]:
        """Partition neural network across available devices."""
        logger.info(f"Partitioning network with {len(available_devices)} devices using {strategy} strategy")
        
        if strategy not in self.partitioning_strategies:
            strategy = 'balanced'
            
        partitions = self.partitioning_strategies[strategy](network_config, available_devices)
        
        # Add inter-partition connections
        self._calculate_inter_partition_connections(partitions, network_config)
        
        logger.info(f"Created {len(partitions)} network partitions")
        return partitions
        
    def _balanced_partitioning(self, network_config: Dict[str, Any], 
                              devices: List[FPGADevice]) -> List[PartitionedNetwork]:
        """Create balanced partitions across devices."""
        num_neurons = network_config.get('neurons', 1000)
        num_devices = len(devices)
        
        partitions = []
        neurons_per_device = num_neurons // num_devices
        remainder = num_neurons % num_devices
        
        current_neuron = 0
        for i, device in enumerate(devices):
            # Distribute remainder across first few devices
            partition_size = neurons_per_device + (1 if i < remainder else 0)
            neuron_indices = list(range(current_neuron, current_neuron + partition_size))
            
            partition = PartitionedNetwork(
                partition_id=f"partition_{i}",
                assigned_device=device.device_id,
                neuron_indices=neuron_indices,
                synapse_connections=[],
                input_mappings={},
                output_mappings={}
            )
            
            partitions.append(partition)
            current_neuron += partition_size
            
        return partitions
        
    def _minimum_cut_partitioning(self, network_config: Dict[str, Any], 
                                 devices: List[FPGADevice]) -> List[PartitionedNetwork]:
        """Partition to minimize inter-device communication."""
        # Simplified minimum cut approach
        # In practice, would use graph partitioning algorithms like METIS
        
        num_neurons = network_config.get('neurons', 1000)
        layers = network_config.get('layers', [])
        
        partitions = []
        
        if layers:
            # Partition by layers to minimize cross-layer communication
            num_devices = len(devices)
            layers_per_device = max(1, len(layers) // num_devices)
            
            current_layer = 0
            neuron_offset = 0
            
            for i, device in enumerate(devices):
                if current_layer >= len(layers):
                    break
                    
                # Assign layers to this device
                device_layers = layers[current_layer:current_layer + layers_per_device]
                neuron_indices = []
                
                for layer in device_layers:
                    layer_size = layer.get('size', 100)
                    neuron_indices.extend(range(neuron_offset, neuron_offset + layer_size))
                    neuron_offset += layer_size
                
                partition = PartitionedNetwork(
                    partition_id=f"partition_{i}",
                    assigned_device=device.device_id,
                    neuron_indices=neuron_indices,
                    synapse_connections=[],
                    input_mappings={},
                    output_mappings={}
                )
                
                partitions.append(partition)
                current_layer += layers_per_device
        else:
            # Fallback to balanced partitioning
            partitions = self._balanced_partitioning(network_config, devices)
            
        return partitions
        
    def _resource_aware_partitioning(self, network_config: Dict[str, Any], 
                                   devices: List[FPGADevice]) -> List[PartitionedNetwork]:
        """Partition based on device resource capabilities."""
        num_neurons = network_config.get('neurons', 1000)
        
        # Calculate resource weights for each device
        total_resources = 0
        device_weights = []
        
        for device in devices:
            # Use primary resource metric as weight
            if 'logic_cells' in device.total_resources:
                weight = device.total_resources['logic_cells']
            elif 'logic_elements' in device.total_resources:
                weight = device.total_resources['logic_elements']
            else:
                weight = device.total_resources.get('generic_units', 1000)
                
            device_weights.append(weight)
            total_resources += weight
            
        # Distribute neurons proportionally to resources
        partitions = []
        current_neuron = 0
        
        for i, (device, weight) in enumerate(zip(devices, device_weights)):
            if i == len(devices) - 1:
                # Last device gets remaining neurons
                partition_size = num_neurons - current_neuron
            else:
                partition_size = int((weight / total_resources) * num_neurons)
                
            neuron_indices = list(range(current_neuron, current_neuron + partition_size))
            
            partition = PartitionedNetwork(
                partition_id=f"partition_{i}",
                assigned_device=device.device_id,
                neuron_indices=neuron_indices,
                synapse_connections=[],
                input_mappings={},
                output_mappings={}
            )
            
            partitions.append(partition)
            current_neuron += partition_size
            
        return partitions
        
    def _latency_optimized_partitioning(self, network_config: Dict[str, Any], 
                                      devices: List[FPGADevice]) -> List[PartitionedNetwork]:
        """Partition to minimize processing latency."""
        # Prioritize devices with better performance history
        sorted_devices = sorted(devices, key=lambda d: d.average_task_time)
        
        # Use balanced partitioning with performance-sorted devices
        return self._balanced_partitioning(network_config, sorted_devices)
        
    def _calculate_inter_partition_connections(self, partitions: List[PartitionedNetwork],
                                             network_config: Dict[str, Any]) -> None:
        """Calculate connections between partitions."""
        connectivity = network_config.get('connectivity', 'sparse_random')
        sparsity = network_config.get('sparsity', 0.1)
        
        # Create partition index mapping
        neuron_to_partition = {}
        for partition in partitions:
            for neuron_idx in partition.neuron_indices:
                neuron_to_partition[neuron_idx] = partition.partition_id
                
        # Generate inter-partition connections based on connectivity pattern
        for partition in partitions:
            for neuron_idx in partition.neuron_indices:
                if connectivity == 'sparse_random':
                    # Generate random connections with specified sparsity
                    total_neurons = sum(len(p.neuron_indices) for p in partitions)
                    num_connections = max(1, int(total_neurons * sparsity))
                    
                    for _ in range(num_connections):
                        target_neuron = np.random.randint(0, total_neurons)
                        target_partition = neuron_to_partition.get(target_neuron)
                        
                        if target_partition and target_partition != partition.partition_id:
                            connection = (target_partition, neuron_idx, target_neuron)
                            if connection not in partition.inter_partition_connections:
                                partition.inter_partition_connections.append(connection)


class FaultToleranceManager:
    """Fault tolerance and recovery management."""
    
    def __init__(self):
        self.device_health = {}
        self.backup_assignments = {}
        self.recovery_strategies = {
            'immediate_failover': self._immediate_failover,
            'graceful_migration': self._graceful_migration,
            'checkpoint_recovery': self._checkpoint_recovery
        }
        
    def monitor_device_health(self, device: FPGADevice) -> bool:
        """Monitor device health and detect failures."""
        current_time = time.time()
        
        # Check heartbeat timeout
        heartbeat_timeout = current_time - device.last_heartbeat
        if heartbeat_timeout > 30.0:  # 30 second timeout
            logger.warning(f"Device {device.device_id} heartbeat timeout: {heartbeat_timeout:.1f}s")
            device.current_state = FPGAState.ERROR
            return False
            
        # Check error rate
        if device.total_tasks_completed > 10:
            error_rate = device.error_count / device.total_tasks_completed
            if error_rate > 0.2:  # 20% error rate threshold
                logger.warning(f"Device {device.device_id} high error rate: {error_rate:.2f}")
                device.current_state = FPGAState.ERROR
                return False
                
        # Check overload condition
        if device.current_load > 95.0:
            logger.warning(f"Device {device.device_id} overloaded: {device.current_load:.1f}%")
            device.current_state = FPGAState.OVERLOADED
            return False
            
        return True
        
    def handle_device_failure(self, failed_device: FPGADevice,
                            available_devices: List[FPGADevice],
                            active_tasks: List[DistributedTask],
                            strategy: str = 'immediate_failover') -> bool:
        """Handle device failure and recovery."""
        logger.error(f"Handling failure for device {failed_device.device_id}")
        
        if strategy not in self.recovery_strategies:
            strategy = 'immediate_failover'
            
        return self.recovery_strategies[strategy](failed_device, available_devices, active_tasks)
        
    def _immediate_failover(self, failed_device: FPGADevice,
                           available_devices: List[FPGADevice],
                           active_tasks: List[DistributedTask]) -> bool:
        """Immediately reassign tasks to other devices."""
        # Find tasks assigned to failed device
        failed_tasks = [task for task in active_tasks 
                       if task.assigned_device == failed_device.device_id]
        
        if not failed_tasks:
            return True
            
        # Find replacement devices
        healthy_devices = [d for d in available_devices 
                          if d.device_id != failed_device.device_id and 
                          d.current_state == FPGAState.READY]
        
        if not healthy_devices:
            logger.error("No healthy devices available for failover")
            return False
            
        load_balancer = LoadBalancer()
        reassigned_count = 0
        
        for task in failed_tasks:
            best_device = load_balancer.select_best_device(healthy_devices, task)
            if best_device:
                task.assigned_device = best_device.device_id
                task.start_time = None  # Reset start time
                reassigned_count += 1
                logger.info(f"Reassigned task {task.task_id} to device {best_device.device_id}")
                
        logger.info(f"Reassigned {reassigned_count}/{len(failed_tasks)} tasks")
        return reassigned_count == len(failed_tasks)
        
    def _graceful_migration(self, failed_device: FPGADevice,
                           available_devices: List[FPGADevice],
                           active_tasks: List[DistributedTask]) -> bool:
        """Gracefully migrate tasks with checkpoint save/restore."""
        # In practice, would save checkpoints before migration
        logger.info("Graceful migration not yet implemented, using immediate failover")
        return self._immediate_failover(failed_device, available_devices, active_tasks)
        
    def _checkpoint_recovery(self, failed_device: FPGADevice,
                            available_devices: List[FPGADevice],
                            active_tasks: List[DistributedTask]) -> bool:
        """Recover tasks from checkpoints."""
        # In practice, would restore from saved checkpoints
        logger.info("Checkpoint recovery not yet implemented, using immediate failover")
        return self._immediate_failover(failed_device, available_devices, active_tasks)


class InterFPGACommunication:
    """Inter-FPGA communication and synchronization."""
    
    def __init__(self):
        self.connections = {}
        self.message_queues = defaultdict(deque)
        self.sync_barriers = {}
        
    async def establish_connection(self, device1: FPGADevice, device2: FPGADevice) -> bool:
        """Establish communication link between two FPGAs."""
        try:
            connection_id = f"{device1.device_id}_{device2.device_id}"
            
            # Simulate connection establishment (would use actual networking)
            reader, writer = await asyncio.open_connection(device2.hostname, device2.port)
            
            self.connections[connection_id] = {
                'device1': device1.device_id,
                'device2': device2.device_id,
                'reader': reader,
                'writer': writer,
                'established_time': time.time(),
                'message_count': 0
            }
            
            logger.info(f"Established connection: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish connection {device1.device_id} -> {device2.device_id}: {e}")
            return False
            
    async def send_message(self, from_device: str, to_device: str, 
                          message: Dict[str, Any]) -> bool:
        """Send message between FPGAs."""
        connection_id = f"{from_device}_{to_device}"
        reverse_connection_id = f"{to_device}_{from_device}"
        
        connection = self.connections.get(connection_id) or self.connections.get(reverse_connection_id)
        
        if not connection:
            logger.error(f"No connection found between {from_device} and {to_device}")
            return False
            
        try:
            # Serialize message
            message_data = json.dumps(message).encode('utf-8')
            message_length = struct.pack('>I', len(message_data))
            
            # Send message (simplified - would use actual network protocol)
            writer = connection['writer']
            writer.write(message_length + message_data)
            await writer.drain()
            
            connection['message_count'] += 1
            logger.debug(f"Sent message from {from_device} to {to_device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {from_device} -> {to_device}: {e}")
            return False
            
    async def broadcast_sync_signal(self, devices: List[str], barrier_id: str) -> None:
        """Broadcast synchronization signal to multiple devices."""
        sync_message = {
            'type': 'sync_barrier',
            'barrier_id': barrier_id,
            'timestamp': time.time()
        }
        
        tasks = []
        for i, device in enumerate(devices):
            for j, other_device in enumerate(devices):
                if i != j:
                    task = self.send_message(device, other_device, sync_message)
                    tasks.append(task)
                    
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Broadcasted sync barrier {barrier_id} to {len(devices)} devices")


class MultiFPGAOrchestrator:
    """Main orchestrator for multi-FPGA distributed processing."""
    
    def __init__(self, orchestrator_id: str = None):
        self.orchestrator_id = orchestrator_id or str(uuid.uuid4())[:8]
        self.devices = {}
        self.active_tasks = {}
        self.task_queue = deque()
        self.completed_tasks = {}
        
        # Components
        self.load_balancer = LoadBalancer()
        self.network_partitioner = NetworkPartitioner()
        self.fault_manager = FaultToleranceManager()
        self.communication = InterFPGACommunication()
        
        # State
        self.running = False
        self.orchestration_thread = None
        self.heartbeat_thread = None
        
        # Statistics
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.average_throughput = 0.0
        
    def register_device(self, device: FPGADevice) -> bool:
        """Register an FPGA device with the orchestrator."""
        try:
            device.last_heartbeat = time.time()
            device.current_state = FPGAState.INITIALIZING
            
            self.devices[device.device_id] = device
            logger.info(f"Registered FPGA device: {device.device_id} ({device.fpga_type})")
            
            # Initialize device state
            device.current_state = FPGAState.READY
            return True
            
        except Exception as e:
            logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
            
    def unregister_device(self, device_id: str) -> bool:
        """Unregister an FPGA device."""
        if device_id in self.devices:
            device = self.devices[device_id]
            device.current_state = FPGAState.OFFLINE
            
            # Handle any active tasks on this device
            active_device_tasks = [task for task in self.active_tasks.values()
                                 if task.assigned_device == device_id]
            
            if active_device_tasks:
                logger.warning(f"Device {device_id} has {len(active_device_tasks)} active tasks")
                # Would handle task migration here
                
            del self.devices[device_id]
            logger.info(f"Unregistered device: {device_id}")
            return True
        else:
            logger.warning(f"Attempted to unregister unknown device: {device_id}")
            return False
            
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a distributed task for processing."""
        task.task_id = task.task_id or str(uuid.uuid4())
        task.created_time = time.time()
        
        # Validate task
        if not self._validate_task(task):
            raise ValueError(f"Invalid task: {task.task_id}")
            
        # Add to task queue
        self.task_queue.append(task)
        logger.info(f"Submitted task: {task.task_id} (priority: {task.priority.name})")
        
        return task.task_id
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task_id,
                'status': 'processing',
                'assigned_device': task.assigned_device,
                'start_time': task.start_time,
                'progress': self._estimate_task_progress(task)
            }
            
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                'task_id': task_id,
                'status': 'completed' if task.result is not None else 'failed',
                'completion_time': task.completion_time,
                'execution_time': task.completion_time - task.start_time if task.start_time else 0,
                'result_available': task.result is not None,
                'error': task.error_message
            }
            
        # Check task queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': 'queued',
                    'queue_position': list(self.task_queue).index(task),
                    'estimated_start_time': self._estimate_start_time(task)
                }
                
        return None
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        healthy_devices = sum(1 for d in self.devices.values() 
                            if d.current_state in [FPGAState.READY, FPGAState.PROCESSING])
        
        total_resources = {}
        available_resources = {}
        
        for device in self.devices.values():
            for resource, amount in device.total_resources.items():
                total_resources[resource] = total_resources.get(resource, 0) + amount
                available_resources[resource] = available_resources.get(resource, 0) + \
                                               device.available_resources.get(resource, 0)
                                               
        return {
            'orchestrator_id': self.orchestrator_id,
            'running': self.running,
            'total_devices': len(self.devices),
            'healthy_devices': healthy_devices,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'total_resources': total_resources,
            'available_resources': available_resources,
            'resource_utilization': {
                resource: 1.0 - (available_resources[resource] / total_resources[resource])
                for resource in total_resources if total_resources[resource] > 0
            },
            'total_tasks_processed': self.total_tasks_processed,
            'average_throughput': self.average_throughput
        }
        
    async def start_orchestration(self) -> None:
        """Start the orchestration system."""
        if self.running:
            logger.warning("Orchestration already running")
            return
            
        self.running = True
        
        # Start orchestration loop
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop)
        self.orchestration_thread.daemon = True
        self.orchestration_thread.start()
        
        # Start heartbeat monitoring
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        logger.info(f"Multi-FPGA orchestrator started: {self.orchestrator_id}")
        
    def stop_orchestration(self) -> None:
        """Stop the orchestration system."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to complete
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=5.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
            
        logger.info("Multi-FPGA orchestrator stopped")
        
    def _validate_task(self, task: DistributedTask) -> bool:
        """Validate task requirements."""
        if not task.network_config:
            logger.error(f"Task {task.task_id} missing network configuration")
            return False
            
        # Check if required resources are available
        if task.resource_requirements:
            total_available = defaultdict(int)
            for device in self.devices.values():
                if device.current_state == FPGAState.READY:
                    for resource, amount in device.available_resources.items():
                        total_available[resource] += amount
                        
            for resource, required in task.resource_requirements.items():
                if total_available[resource] < required:
                    logger.error(f"Insufficient {resource}: need {required}, have {total_available[resource]}")
                    return False
                    
        return True
        
    def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self.running:
            try:
                # Process task queue
                self._process_task_queue()
                
                # Monitor active tasks
                self._monitor_active_tasks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Small sleep to prevent tight loop
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                time.sleep(1.0)
                
    def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring loop."""
        while self.running:
            try:
                current_time = time.time()
                
                for device in list(self.devices.values()):
                    # Simulate heartbeat reception (would come from actual devices)
                    if device.current_state != FPGAState.OFFLINE:
                        device.last_heartbeat = current_time
                        
                    # Check device health
                    is_healthy = self.fault_manager.monitor_device_health(device)
                    
                    if not is_healthy:
                        # Handle device failure
                        available_devices = [d for d in self.devices.values() 
                                           if d.device_id != device.device_id]
                        active_tasks = list(self.active_tasks.values())
                        
                        self.fault_manager.handle_device_failure(
                            device, available_devices, active_tasks
                        )
                        
                time.sleep(5.0)  # Heartbeat check every 5 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(5.0)
                
    def _process_task_queue(self) -> None:
        """Process pending tasks in the queue."""
        if not self.task_queue:
            return
            
        # Sort queue by priority
        sorted_queue = sorted(self.task_queue, key=lambda t: t.priority.value, reverse=True)
        self.task_queue = deque(sorted_queue)
        
        # Process high-priority tasks first
        tasks_to_remove = []
        
        for task in list(self.task_queue):
            # Find suitable device
            available_devices = [d for d in self.devices.values() 
                               if d.current_state in [FPGAState.READY, FPGAState.PROCESSING]]
            
            best_device = self.load_balancer.select_best_device(available_devices, task)
            
            if best_device:
                # Assign task to device
                task.assigned_device = best_device.device_id
                task.start_time = time.time()
                
                # Move to active tasks
                self.active_tasks[task.task_id] = task
                tasks_to_remove.append(task)
                
                # Update device state
                best_device.current_state = FPGAState.PROCESSING
                best_device.task_queue_size += 1
                
                logger.info(f"Assigned task {task.task_id} to device {best_device.device_id}")
                
                # Start task processing (would send to actual FPGA)
                self._start_task_processing(task, best_device)
                
        # Remove assigned tasks from queue
        for task in tasks_to_remove:
            self.task_queue.remove(task)
            
    def _monitor_active_tasks(self) -> None:
        """Monitor active task execution."""
        current_time = time.time()
        completed_tasks = []
        
        for task_id, task in list(self.active_tasks.items()):
            # Check for timeout
            if task.start_time and (current_time - task.start_time) > task.max_execution_time:
                logger.warning(f"Task {task_id} timed out")
                task.error_message = "Execution timeout"
                task.completion_time = current_time
                completed_tasks.append(task_id)
                continue
                
            # Simulate task completion (would receive from actual FPGA)
            if task.start_time and (current_time - task.start_time) > 10.0:  # Simulate 10s processing
                task.result = f"Simulated result for {task_id}"
                task.completion_time = current_time
                completed_tasks.append(task_id)
                
        # Move completed tasks
        for task_id in completed_tasks:
            task = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = task
            
            # Update device state
            if task.assigned_device in self.devices:
                device = self.devices[task.assigned_device]
                device.task_queue_size = max(0, device.task_queue_size - 1)
                device.total_tasks_completed += 1
                
                if task.completion_time and task.start_time:
                    execution_time = task.completion_time - task.start_time
                    device.average_task_time = (
                        (device.average_task_time * (device.total_tasks_completed - 1) + execution_time) /
                        device.total_tasks_completed
                    )
                    
                if task.error_message:
                    device.error_count += 1
                    
                # Set device back to ready if no more tasks
                if device.task_queue_size == 0:
                    device.current_state = FPGAState.READY
                    
            self.total_tasks_processed += 1
            logger.info(f"Task {task_id} completed")
            
    def _start_task_processing(self, task: DistributedTask, device: FPGADevice) -> None:
        """Start task processing on assigned device."""
        # In practice, would send task to actual FPGA device
        # This is a simulation placeholder
        logger.debug(f"Starting task {task.task_id} on device {device.device_id}")
        
    def _estimate_task_progress(self, task: DistributedTask) -> float:
        """Estimate task completion progress."""
        if not task.start_time:
            return 0.0
            
        elapsed_time = time.time() - task.start_time
        estimated_total_time = task.max_execution_time * 0.5  # Assume average 50% of max time
        
        progress = min(elapsed_time / estimated_total_time, 0.99)
        return progress
        
    def _estimate_start_time(self, task: DistributedTask) -> float:
        """Estimate when a queued task will start."""
        # Simple estimation based on queue position and average task time
        queue_position = list(self.task_queue).index(task)
        average_task_time = 10.0  # Simplified average
        
        return time.time() + queue_position * average_task_time
        
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        if self.total_tasks_processed > 0 and self.total_processing_time > 0:
            self.average_throughput = self.total_tasks_processed / self.total_processing_time
        else:
            self.average_throughput = 0.0


# Convenience functions for easy usage

def create_orchestrator(orchestrator_id: str = None) -> MultiFPGAOrchestrator:
    """Create a new multi-FPGA orchestrator."""
    return MultiFPGAOrchestrator(orchestrator_id)


def create_fpga_device(device_id: str, hostname: str, fpga_type: str, 
                      port: int = 8080) -> FPGADevice:
    """Create a new FPGA device configuration."""
    return FPGADevice(
        device_id=device_id,
        hostname=hostname,
        port=port,
        fpga_type=fpga_type,
        capabilities={}
    )


def create_distributed_task(task_type: str, network_config: Dict[str, Any],
                           priority: TaskPriority = TaskPriority.NORMAL,
                           **kwargs) -> DistributedTask:
    """Create a new distributed task."""
    return DistributedTask(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        priority=priority,
        network_config=network_config,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Create orchestrator
        orchestrator = create_orchestrator("test_orchestrator")
        
        # Create mock FPGA devices
        devices = [
            create_fpga_device("fpga_001", "192.168.1.10", "artix7_35t"),
            create_fpga_device("fpga_002", "192.168.1.11", "artix7_100t"),
            create_fpga_device("fpga_003", "192.168.1.12", "cyclone5_gx")
        ]
        
        # Register devices
        for device in devices:
            orchestrator.register_device(device)
            
        # Start orchestration
        await orchestrator.start_orchestration()
        
        # Create test tasks
        network_config = {
            'neurons': 1000,
            'layers': [
                {'type': 'input', 'size': 100},
                {'type': 'hidden', 'size': 800},
                {'type': 'output', 'size': 100}
            ],
            'connectivity': 'sparse_random',
            'sparsity': 0.1
        }
        
        tasks = []
        for i in range(5):
            task = create_distributed_task(
                task_type="snn_inference",
                network_config=network_config,
                priority=TaskPriority.NORMAL
            )
            task_id = orchestrator.submit_task(task)
            tasks.append(task_id)
            
        print(f"Submitted {len(tasks)} tasks")
        
        # Monitor system
        for _ in range(30):  # Monitor for 30 seconds
            status = orchestrator.get_system_status()
            print(f"System Status: {status['healthy_devices']}/{status['total_devices']} devices, "
                  f"{status['active_tasks']} active tasks, {status['completed_tasks']} completed")
            
            await asyncio.sleep(1)
            
        # Stop orchestration
        orchestrator.stop_orchestration()
        
    # Run example
    asyncio.run(main())