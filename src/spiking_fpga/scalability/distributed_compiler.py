"""Distributed compilation system with intelligent workload balancing."""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import pickle
import tempfile
import shutil

from .auto_scaler import AutoScaler, ScalingConfig
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


@dataclass
class CompilationTask:
    """Represents a compilation task that can be distributed."""
    task_id: str
    network_config: Dict[str, Any]
    target: str
    optimization_level: int
    options: Dict[str, Any]
    priority: int = 1
    estimated_duration: float = 0.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CompilationResult:
    """Result of a distributed compilation task."""
    task_id: str
    success: bool
    duration: float
    output_files: Dict[str, Path]
    errors: List[str] = None
    warnings: List[str] = None
    worker_id: str = None
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.resource_usage is None:
            self.resource_usage = {}


class DistributedCompiler:
    """High-performance distributed compilation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Auto-scaling configuration
        scaling_config = ScalingConfig(
            min_workers=self.config.get('min_workers', 2),
            max_workers=self.config.get('max_workers', 16),
            target_cpu_usage=self.config.get('target_cpu_usage', 70.0),
            scale_up_threshold=self.config.get('scale_up_threshold', 80.0),
            scale_down_threshold=self.config.get('scale_down_threshold', 40.0)
        )
        
        self.auto_scaler = AutoScaler(scaling_config)
        self.load_balancer = LoadBalancer()
        
        # Task management
        self.pending_tasks: Dict[str, CompilationTask] = {}
        self.running_tasks: Dict[str, CompilationTask] = {}
        self.completed_tasks: Dict[str, CompilationResult] = {}
        
        # Caching and optimization
        self.cache_dir = Path(tempfile.gettempdir()) / "spiking_fpga_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_enabled = self.config.get('enable_cache', True)
        
        # Performance tracking
        self.compilation_stats: Dict[str, List[float]] = {
            'durations': [],
            'throughput': [],
            'cache_hits': [],
            'error_rates': []
        }
        
        self._shutdown = False
        self._stats_lock = threading.Lock()
        
    def start(self):
        """Start the distributed compilation system."""
        self.auto_scaler.start()
        logger.info("Distributed compiler started")
    
    def stop(self):
        """Stop the distributed compilation system."""
        self._shutdown = True
        self.auto_scaler.stop()
        logger.info("Distributed compiler stopped")
    
    def submit_compilation(self, network_config: Dict[str, Any], 
                          target: str, **options) -> str:
        """Submit a compilation task and return task ID."""
        # Generate unique task ID
        task_content = json.dumps({
            'network': network_config,
            'target': target,
            'options': options
        }, sort_keys=True)
        task_id = hashlib.sha256(task_content.encode()).hexdigest()[:16]
        
        # Check cache first
        if self.cache_enabled:
            cached_result = self._check_cache(task_id)
            if cached_result:
                logger.info(f"Cache hit for task {task_id}")
                self.completed_tasks[task_id] = cached_result
                return task_id
        
        # Create compilation task
        task = CompilationTask(
            task_id=task_id,
            network_config=network_config,
            target=target,
            optimization_level=options.get('optimization_level', 2),
            options=options,
            priority=options.get('priority', 1),
            estimated_duration=self._estimate_compilation_time(network_config, target)
        )
        
        # Add dependencies if specified
        if 'depends_on' in options:
            task.dependencies = options['depends_on']
        
        self.pending_tasks[task_id] = task
        
        # Schedule for execution
        self._schedule_task(task)
        
        logger.info(f"Submitted compilation task {task_id}")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> CompilationResult:
        """Get compilation result (blocking until complete)."""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            if self._shutdown:
                raise RuntimeError("Distributed compiler is shutting down")
            
            time.sleep(0.1)
    
    def get_status(self, task_id: str) -> str:
        """Get current status of a task."""
        if task_id in self.completed_tasks:
            return 'completed'
        elif task_id in self.running_tasks:
            return 'running'
        elif task_id in self.pending_tasks:
            return 'pending'
        else:
            return 'unknown'
    
    def list_tasks(self) -> Dict[str, str]:
        """List all tasks and their statuses."""
        tasks = {}
        
        for task_id in self.pending_tasks:
            tasks[task_id] = 'pending'
        for task_id in self.running_tasks:
            tasks[task_id] = 'running'
        for task_id in self.completed_tasks:
            tasks[task_id] = 'completed'
        
        return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]
            logger.info(f"Cancelled pending task {task_id}")
            return True
        elif task_id in self.running_tasks:
            # Mark for cancellation (worker will check this)
            self.running_tasks[task_id].options['cancelled'] = True
            logger.info(f"Marked running task {task_id} for cancellation")
            return True
        else:
            return False
    
    def batch_compile(self, tasks: List[Dict[str, Any]], 
                     max_parallel: Optional[int] = None) -> List[str]:
        """Submit multiple compilation tasks for batch processing."""
        task_ids = []
        
        # Submit all tasks
        for task_config in tasks:
            network_config = task_config['network_config']
            target = task_config['target']
            options = task_config.get('options', {})
            
            task_id = self.submit_compilation(network_config, target, **options)
            task_ids.append(task_id)
        
        # Set up dependencies for ordered execution if needed
        if task_config.get('sequential', False):
            for i in range(1, len(task_ids)):
                self.pending_tasks[task_ids[i]].dependencies = [task_ids[i-1]]
        
        logger.info(f"Submitted batch of {len(task_ids)} compilation tasks")
        return task_ids
    
    def wait_for_batch(self, task_ids: List[str], 
                      timeout: Optional[float] = None) -> List[CompilationResult]:
        """Wait for batch completion and return all results."""
        results = []
        
        for task_id in task_ids:
            try:
                result = self.get_result(task_id, timeout)
                results.append(result)
            except TimeoutError:
                logger.warning(f"Task {task_id} timed out")
                # Create a timeout result
                results.append(CompilationResult(
                    task_id=task_id,
                    success=False,
                    duration=timeout or 0,
                    output_files={},
                    errors=['Task timed out']
                ))
        
        return results
    
    def _schedule_task(self, task: CompilationTask):
        """Schedule a task for execution with load balancing."""
        def execute_task():
            # Check dependencies
            if not self._check_dependencies(task):
                # Re-schedule after a delay
                threading.Timer(5.0, lambda: self._schedule_task(task)).start()
                return
            
            # Move task from pending to running
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
                self.running_tasks[task.task_id] = task
            
            # Execute the actual compilation
            result = self._execute_compilation(task)
            
            # Move task from running to completed
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
                self.completed_tasks[task.task_id] = result
            
            # Cache result if successful
            if self.cache_enabled and result.success:
                self._cache_result(task.task_id, result)
            
            # Update statistics
            self._update_stats(result)
        
        # Submit to auto-scaler for execution
        self.auto_scaler.submit_task(execute_task)
    
    def _execute_compilation(self, task: CompilationTask) -> CompilationResult:
        """Execute a single compilation task."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        logger.info(f"Starting compilation task {task.task_id} on worker {worker_id}")
        
        try:
            # Check for cancellation
            if task.options.get('cancelled', False):
                return CompilationResult(
                    task_id=task.task_id,
                    success=False,
                    duration=time.time() - start_time,
                    output_files={},
                    errors=['Task was cancelled'],
                    worker_id=worker_id
                )
            
            # Create temporary workspace
            workspace = self.cache_dir / f"task_{task.task_id}"
            workspace.mkdir(exist_ok=True)
            
            try:
                # Import here to avoid circular imports
                from spiking_fpga.network_compiler import compile_network
                from spiking_fpga.core import FPGATarget
                from spiking_fpga.models.optimization import OptimizationLevel
                
                # Convert config to network object
                network_file = workspace / "network.yaml"
                with open(network_file, 'w') as f:
                    json.dump(task.network_config, f, indent=2)
                
                # Set up compilation options
                fpga_target = FPGATarget(task.target)
                opt_level = OptimizationLevel(task.optimization_level)
                output_dir = workspace / "output"
                
                # Execute compilation
                result = compile_network(
                    network=network_file,
                    target=fpga_target,
                    output_dir=output_dir,
                    optimization_level=opt_level,
                    **task.options
                )
                
                # Collect output files
                output_files = {}
                if output_dir.exists():
                    for file_path in output_dir.rglob('*'):
                        if file_path.is_file():
                            output_files[file_path.name] = file_path
                
                compilation_result = CompilationResult(
                    task_id=task.task_id,
                    success=result.success,
                    duration=time.time() - start_time,
                    output_files=output_files,
                    errors=result.errors if hasattr(result, 'errors') else [],
                    warnings=result.warnings if hasattr(result, 'warnings') else [],
                    worker_id=worker_id,
                    resource_usage=self._get_resource_usage()
                )
                
                logger.info(f"Completed task {task.task_id} in {compilation_result.duration:.2f}s")
                return compilation_result
                
            finally:
                # Cleanup workspace
                if workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Compilation task {task.task_id} failed: {e}")
            return CompilationResult(
                task_id=task.task_id,
                success=False,
                duration=time.time() - start_time,
                output_files={},
                errors=[str(e)],
                worker_id=worker_id
            )
    
    def _check_dependencies(self, task: CompilationTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep_id].success:
                # Dependency failed, fail this task too
                result = CompilationResult(
                    task_id=task.task_id,
                    success=False,
                    duration=0,
                    output_files={},
                    errors=[f'Dependency {dep_id} failed']
                )
                self.completed_tasks[task.task_id] = result
                return False
        return True
    
    def _estimate_compilation_time(self, network_config: Dict[str, Any], 
                                  target: str) -> float:
        """Estimate compilation time based on network complexity and historical data."""
        # Simple heuristic based on network size
        total_neurons = 0
        total_synapses = 0
        
        if 'layers' in network_config:
            for layer in network_config['layers']:
                total_neurons += layer.get('size', 0)
        
        if 'connections' in network_config:
            total_synapses = len(network_config['connections']) * 1000  # Estimate
        
        # Base time + complexity factors
        base_time = 30.0  # 30 seconds base
        neuron_factor = total_neurons * 0.001  # 1ms per neuron
        synapse_factor = total_synapses * 0.0001  # 0.1ms per synapse
        
        estimated = base_time + neuron_factor + synapse_factor
        
        # Adjust based on target complexity
        target_multipliers = {
            'artix7_35t': 1.0,
            'artix7_100t': 1.2,
            'cyclone5_gx': 1.1,
            'cyclone5_gt': 1.3
        }
        
        return estimated * target_multipliers.get(target, 1.0)
    
    def _check_cache(self, task_id: str) -> Optional[CompilationResult]:
        """Check if compilation result is cached."""
        cache_file = self.cache_dir / f"result_{task_id}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Verify cache entry is recent (less than 1 hour old)
                if time.time() - cache_file.stat().st_mtime < 3600:
                    with self._stats_lock:
                        self.compilation_stats['cache_hits'].append(1)
                    return result
                else:
                    # Remove stale cache entry
                    cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to load cache for {task_id}: {e}")
                
        return None
    
    def _cache_result(self, task_id: str, result: CompilationResult):
        """Cache compilation result for future use."""
        cache_file = self.cache_dir / f"result_{task_id}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result for {task_id}: {e}")
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            return {}
    
    def _update_stats(self, result: CompilationResult):
        """Update compilation statistics."""
        with self._stats_lock:
            self.compilation_stats['durations'].append(result.duration)
            self.compilation_stats['error_rates'].append(0 if result.success else 1)
            
            # Keep only recent stats
            for key in self.compilation_stats:
                if len(self.compilation_stats[key]) > 1000:
                    self.compilation_stats[key] = self.compilation_stats[key][-500:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._stats_lock:
            if not self.compilation_stats['durations']:
                return {}
            
            durations = self.compilation_stats['durations']
            error_rates = self.compilation_stats['error_rates']
            
            return {
                'total_compilations': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'success_rate': (1 - sum(error_rates) / len(error_rates)) * 100,
                'cache_hit_rate': (sum(self.compilation_stats['cache_hits']) / 
                                 max(len(durations), 1)) * 100,
                'pending_tasks': len(self.pending_tasks),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'auto_scaler_stats': self.auto_scaler.get_statistics()
            }