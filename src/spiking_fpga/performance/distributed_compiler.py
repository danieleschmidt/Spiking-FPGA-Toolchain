"""
Distributed compilation system for massive scalability.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

from ..network_compiler import NetworkCompiler, CompilationResult, CompilationConfig
from ..core import FPGATarget
from ..reliability import FaultTolerantCompiler, FaultToleranceConfig

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Compilation node status."""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class CompilationNode:
    """Represents a compilation node in the distributed system."""
    node_id: str
    hostname: str
    targets: List[FPGATarget]
    max_concurrent_jobs: int = 4
    current_jobs: int = 0
    status: NodeStatus = NodeStatus.IDLE
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = 0
    total_compilations: int = 0
    successful_compilations: int = 0
    average_completion_time: float = 0.0


@dataclass
class CompilationJob:
    """Represents a compilation job in the distributed system."""
    job_id: str
    network: Any
    output_dir: Path
    config: CompilationConfig
    target: FPGATarget
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[CompilationResult] = None
    retry_count: int = 0


@dataclass
class DistributedConfig:
    """Configuration for distributed compilation."""
    max_retries: int = 3
    job_timeout: float = 3600.0  # 1 hour
    heartbeat_interval: float = 30.0  # 30 seconds
    load_balancing_strategy: str = "least_loaded"  # least_loaded, round_robin, performance_based
    enable_job_stealing: bool = True
    enable_adaptive_scheduling: bool = True
    failure_threshold: float = 0.1  # 10% failure rate triggers node quarantine


class DistributedCompiler:
    """
    Distributed compilation system for massive scalability.
    
    Features:
    - Automatic load balancing across multiple compilation nodes
    - Fault-tolerant job scheduling with retry mechanisms
    - Real-time performance monitoring and optimization
    - Adaptive resource allocation based on workload patterns
    - Job stealing for optimal resource utilization
    """
    
    def __init__(self, config: DistributedConfig = None):
        self.config = config or DistributedConfig()
        self.nodes: Dict[str, CompilationNode] = {}
        self.job_queue: List[CompilationJob] = []
        self.active_jobs: Dict[str, CompilationJob] = {}
        self.completed_jobs: Dict[str, CompilationJob] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        self._job_queue_condition = threading.Condition(self._lock)
        
        # Background threads
        self._scheduler_thread = None
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.cluster_metrics = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'average_completion_time': 0.0,
            'throughput_jobs_per_hour': 0.0
        }
        
        # Workload balancer
        self.workload_balancer = WorkloadBalancer()
        
        # Start background services
        self._start_background_services()
        
        logger.info("DistributedCompiler initialized")
    
    def add_node(self, node: CompilationNode):
        """Add a compilation node to the cluster."""
        with self._lock:
            self.nodes[node.node_id] = node
            node.last_heartbeat = time.time()
            logger.info(f"Added compilation node: {node.node_id} with targets {[t.value for t in node.targets]}")
    
    def remove_node(self, node_id: str):
        """Remove a compilation node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                # Reschedule any active jobs from this node
                jobs_to_reschedule = [
                    job for job in self.active_jobs.values() 
                    if job.assigned_node == node_id
                ]
                
                for job in jobs_to_reschedule:
                    job.assigned_node = None
                    job.started_at = None
                    job.retry_count += 1
                    self.job_queue.append(job)
                    del self.active_jobs[job.job_id]
                
                del self.nodes[node_id]
                logger.warning(f"Removed node {node_id}, rescheduled {len(jobs_to_reschedule)} jobs")
    
    def submit_job(self, network: Any, output_dir: Path, config: CompilationConfig, 
                   target: FPGATarget, priority: int = 0) -> str:
        """Submit a compilation job to the distributed system."""
        job_id = self._generate_job_id(network, config)
        
        job = CompilationJob(
            job_id=job_id,
            network=network,
            output_dir=output_dir,
            config=config,
            target=target,
            priority=priority
        )
        
        with self._job_queue_condition:
            self.job_queue.append(job)
            self.cluster_metrics['total_jobs'] += 1
            self._job_queue_condition.notify()
        
        logger.info(f"Submitted job {job_id} with target {target.value}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a compilation job."""
        with self._lock:
            # Check active jobs
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return {
                    'job_id': job_id,
                    'status': 'running',
                    'assigned_node': job.assigned_node,
                    'started_at': job.started_at,
                    'elapsed_time': time.time() - job.started_at if job.started_at else 0
                }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return {
                    'job_id': job_id,
                    'status': 'completed' if job.result and job.result.success else 'failed',
                    'assigned_node': job.assigned_node,
                    'completed_at': job.completed_at,
                    'duration': job.completed_at - job.started_at if job.started_at and job.completed_at else 0,
                    'result': job.result
                }
            
            # Check queued jobs
            for job in self.job_queue:
                if job.job_id == job_id:
                    return {
                        'job_id': job_id,
                        'status': 'queued',
                        'created_at': job.created_at,
                        'queue_time': time.time() - job.created_at
                    }
        
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'status': node.status.value,
                    'current_jobs': node.current_jobs,
                    'max_concurrent_jobs': node.max_concurrent_jobs,
                    'utilization': node.current_jobs / node.max_concurrent_jobs if node.max_concurrent_jobs > 0 else 0,
                    'total_compilations': node.total_compilations,
                    'success_rate': node.successful_compilations / node.total_compilations if node.total_compilations > 0 else 0,
                    'avg_completion_time': node.average_completion_time,
                    'last_heartbeat': node.last_heartbeat
                }
            
            return {
                'total_nodes': len(self.nodes),
                'active_nodes': len([n for n in self.nodes.values() if n.status != NodeStatus.FAILED]),
                'total_jobs_queued': len(self.job_queue),
                'total_jobs_active': len(self.active_jobs),
                'total_jobs_completed': len(self.completed_jobs),
                'cluster_metrics': self.cluster_metrics,
                'node_details': node_stats
            }
    
    def _start_background_services(self):
        """Start background scheduler and monitor threads."""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="DistributedScheduler"
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="DistributedMonitor"
        )
        
        self._scheduler_thread.start()
        self._monitor_thread.start()
        
        logger.info("Background services started")
    
    def _scheduler_loop(self):
        """Main scheduler loop for job assignment."""
        while not self._stop_event.is_set():
            try:
                with self._job_queue_condition:
                    while not self.job_queue and not self._stop_event.is_set():
                        self._job_queue_condition.wait(timeout=1.0)
                    
                    if self._stop_event.is_set():
                        break
                    
                    # Process queued jobs
                    jobs_to_schedule = []
                    for job in self.job_queue[:]:
                        if job.retry_count > self.config.max_retries:
                            logger.error(f"Job {job.job_id} exceeded max retries, marking as failed")
                            job.result = self._create_failure_result(job, "Max retries exceeded")
                            self._mark_job_completed(job)
                            self.job_queue.remove(job)
                            continue
                        
                        node = self._find_best_node_for_job(job)
                        if node:
                            job.assigned_node = node.node_id
                            job.started_at = time.time()
                            node.current_jobs += 1
                            node.status = NodeStatus.BUSY if node.current_jobs < node.max_concurrent_jobs else NodeStatus.OVERLOADED
                            
                            self.active_jobs[job.job_id] = job
                            jobs_to_schedule.append(job)
                            self.job_queue.remove(job)
                            
                            logger.info(f"Assigned job {job.job_id} to node {node.node_id}")
                    
                    # Start jobs on their assigned nodes
                    for job in jobs_to_schedule:
                        self._execute_job_async(job)
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
    
    def _monitor_loop(self):
        """Monitor node health and job timeouts."""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check node health
                with self._lock:
                    dead_nodes = []
                    for node_id, node in self.nodes.items():
                        if current_time - node.last_heartbeat > self.config.heartbeat_interval * 3:
                            logger.warning(f"Node {node_id} appears to be dead (last heartbeat: {current_time - node.last_heartbeat:.1f}s ago)")
                            node.status = NodeStatus.FAILED
                            dead_nodes.append(node_id)
                    
                    # Remove dead nodes
                    for node_id in dead_nodes:
                        self.remove_node(node_id)
                    
                    # Check job timeouts
                    timed_out_jobs = []
                    for job_id, job in self.active_jobs.items():
                        if job.started_at and (current_time - job.started_at) > self.config.job_timeout:
                            logger.warning(f"Job {job_id} timed out after {current_time - job.started_at:.1f}s")
                            timed_out_jobs.append(job)
                    
                    # Handle timed out jobs
                    for job in timed_out_jobs:
                        self._handle_job_timeout(job)
                
                # Update cluster metrics
                self._update_cluster_metrics()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _find_best_node_for_job(self, job: CompilationJob) -> Optional[CompilationNode]:
        """Find the best node to execute a job."""
        available_nodes = [
            node for node in self.nodes.values()
            if (node.status in [NodeStatus.IDLE, NodeStatus.BUSY] and
                node.current_jobs < node.max_concurrent_jobs and
                job.target in node.targets)
        ]
        
        if not available_nodes:
            return None
        
        # Apply load balancing strategy
        if self.config.load_balancing_strategy == "least_loaded":
            return min(available_nodes, key=lambda n: n.current_jobs / n.max_concurrent_jobs)
        elif self.config.load_balancing_strategy == "performance_based":
            return min(available_nodes, key=lambda n: n.average_completion_time or float('inf'))
        elif self.config.load_balancing_strategy == "round_robin":
            return self.workload_balancer.get_next_node(available_nodes)
        else:
            return available_nodes[0]
    
    def _execute_job_async(self, job: CompilationJob):
        """Execute job asynchronously on assigned node."""
        def run_job():
            try:
                node = self.nodes.get(job.assigned_node)
                if not node:
                    raise RuntimeError(f"Assigned node {job.assigned_node} not found")
                
                # Create compiler for the target
                compiler = NetworkCompiler(job.target, enable_monitoring=True)
                
                # Execute compilation
                result = compiler.compile(job.network, job.output_dir, job.config)
                
                # Update job result
                job.result = result
                job.completed_at = time.time()
                
                # Update node metrics
                completion_time = job.completed_at - job.started_at
                node.total_compilations += 1
                if result.success:
                    node.successful_compilations += 1
                
                # Update average completion time using exponential moving average
                alpha = 0.1
                if node.average_completion_time == 0:
                    node.average_completion_time = completion_time
                else:
                    node.average_completion_time = (
                        alpha * completion_time + (1 - alpha) * node.average_completion_time
                    )
                
                logger.info(f"Job {job.job_id} completed {'successfully' if result.success else 'with errors'} in {completion_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Job {job.job_id} failed with exception: {e}")
                job.result = self._create_failure_result(job, str(e))
                job.completed_at = time.time()
            
            finally:
                # Clean up
                self._mark_job_completed(job)
        
        # Execute in thread pool
        threading.Thread(target=run_job, daemon=True).start()
    
    def _mark_job_completed(self, job: CompilationJob):
        """Mark job as completed and update metrics."""
        with self._lock:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            self.completed_jobs[job.job_id] = job
            
            # Update node status
            if job.assigned_node and job.assigned_node in self.nodes:
                node = self.nodes[job.assigned_node]
                node.current_jobs = max(0, node.current_jobs - 1)
                if node.current_jobs == 0:
                    node.status = NodeStatus.IDLE
                elif node.current_jobs < node.max_concurrent_jobs:
                    node.status = NodeStatus.BUSY
            
            # Update cluster metrics
            if job.result and job.result.success:
                self.cluster_metrics['successful_jobs'] += 1
            else:
                self.cluster_metrics['failed_jobs'] += 1
            
            # Limit completed jobs history
            if len(self.completed_jobs) > 1000:
                oldest_jobs = sorted(self.completed_jobs.items(), 
                                   key=lambda x: x[1].completed_at or 0)[:100]
                for job_id, _ in oldest_jobs:
                    del self.completed_jobs[job_id]
    
    def _handle_job_timeout(self, job: CompilationJob):
        """Handle job timeout."""
        job.result = self._create_failure_result(job, "Job timeout")
        job.completed_at = time.time()
        self._mark_job_completed(job)
        
        # Potentially quarantine the node if it has too many timeouts
        if job.assigned_node and job.assigned_node in self.nodes:
            node = self.nodes[job.assigned_node]
            failure_rate = (node.total_compilations - node.successful_compilations) / max(1, node.total_compilations)
            
            if failure_rate > self.config.failure_threshold:
                logger.warning(f"Node {job.assigned_node} quarantined due to high failure rate: {failure_rate:.2%}")
                node.status = NodeStatus.MAINTENANCE
    
    def _create_failure_result(self, job: CompilationJob, error_message: str) -> CompilationResult:
        """Create a failure compilation result."""
        from ..models.network import Network
        from ..models.optimization import ResourceEstimate
        
        return CompilationResult(
            success=False,
            network=Network(name="failed"),
            optimized_network=Network(name="failed"), 
            hdl_files={},
            resource_estimate=ResourceEstimate(),
            optimization_stats={},
            errors=[error_message]
        )
    
    def _update_cluster_metrics(self):
        """Update cluster performance metrics."""
        with self._lock:
            completed_count = len(self.completed_jobs)
            if completed_count == 0:
                return
            
            # Calculate average completion time
            recent_jobs = list(self.completed_jobs.values())[-100:]  # Last 100 jobs
            completion_times = [
                job.completed_at - job.started_at
                for job in recent_jobs
                if job.started_at and job.completed_at
            ]
            
            if completion_times:
                self.cluster_metrics['average_completion_time'] = sum(completion_times) / len(completion_times)
            
            # Calculate throughput (jobs per hour)
            if len(recent_jobs) >= 2:
                time_window = recent_jobs[-1].completed_at - recent_jobs[0].completed_at
                if time_window > 0:
                    self.cluster_metrics['throughput_jobs_per_hour'] = len(recent_jobs) * 3600 / time_window
    
    def _generate_job_id(self, network, config) -> str:
        """Generate unique job ID."""
        content = f"{network}_{config}_{time.time()}"
        return f"job_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def node_heartbeat(self, node_id: str, performance_data: Dict[str, Any] = None):
        """Process heartbeat from a node."""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                if performance_data:
                    node.performance_metrics.update(performance_data)
                
                # Auto-recover from maintenance mode if node is healthy
                if node.status == NodeStatus.MAINTENANCE:
                    failure_rate = (node.total_compilations - node.successful_compilations) / max(1, node.total_compilations)
                    if failure_rate < self.config.failure_threshold / 2:  # Hysteresis
                        node.status = NodeStatus.IDLE
                        logger.info(f"Node {node_id} recovered from maintenance mode")
    
    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[CompilationResult]:
        """Wait for a job to complete and return its result."""
        start_time = time.time()
        
        while True:
            status = self.get_job_status(job_id)
            if not status:
                return None
            
            if status['status'] in ['completed', 'failed']:
                return status.get('result')
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(1.0)
    
    def shutdown(self):
        """Shutdown the distributed compiler."""
        logger.info("Shutting down distributed compiler...")
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Distributed compiler shutdown complete")


class WorkloadBalancer:
    """Implements various load balancing strategies."""
    
    def __init__(self):
        self.round_robin_index = 0
    
    def get_next_node(self, available_nodes: List[CompilationNode]) -> CompilationNode:
        """Get next node using round-robin strategy."""
        if not available_nodes:
            raise ValueError("No available nodes")
        
        node = available_nodes[self.round_robin_index % len(available_nodes)]
        self.round_robin_index += 1
        return node


class ClusterManager:
    """Manages a cluster of compilation nodes."""
    
    def __init__(self, distributed_compiler: DistributedCompiler):
        self.compiler = distributed_compiler
        self.auto_discovery_enabled = False
        self._discovery_thread = None
    
    def auto_discover_nodes(self, network_range: str = "192.168.1.0/24"):
        """Auto-discover compilation nodes on the network."""
        # This would implement network discovery logic
        logger.info(f"Auto-discovery not yet implemented for range: {network_range}")
    
    def add_local_node(self, targets: List[FPGATarget], max_jobs: int = 4) -> str:
        """Add the local machine as a compilation node."""
        import socket
        hostname = socket.gethostname()
        node_id = f"local_{hostname}"
        
        node = CompilationNode(
            node_id=node_id,
            hostname=hostname,
            targets=targets,
            max_concurrent_jobs=max_jobs
        )
        
        self.compiler.add_node(node)
        return node_id
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get comprehensive cluster health metrics."""
        status = self.compiler.get_cluster_status()
        
        # Add health analysis
        total_capacity = sum(node['max_concurrent_jobs'] for node in status['node_details'].values())
        current_utilization = sum(node['current_jobs'] for node in status['node_details'].values())
        
        health_score = 100.0
        
        # Reduce score based on failed nodes
        if status['total_nodes'] > 0:
            active_ratio = status['active_nodes'] / status['total_nodes']
            health_score *= active_ratio
        
        # Reduce score based on high utilization
        if total_capacity > 0:
            utilization_ratio = current_utilization / total_capacity
            if utilization_ratio > 0.8:  # High utilization threshold
                health_score *= (1.0 - (utilization_ratio - 0.8) * 2)
        
        return {
            'health_score': max(0, health_score),
            'total_capacity': total_capacity,
            'current_utilization': current_utilization,
            'utilization_percentage': (current_utilization / total_capacity * 100) if total_capacity > 0 else 0,
            'cluster_status': status
        }