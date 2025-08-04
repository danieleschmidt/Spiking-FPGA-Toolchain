"""Concurrent processing and resource pooling for FPGA compilation."""

import asyncio
import concurrent.futures
import threading
import multiprocessing
import queue
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import psutil

from .logging import StructuredLogger

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TaskResult:
    """Result of a concurrent task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    worker_id: Optional[str] = None


class ResourcePool(Generic[T]):
    """Thread-safe resource pool for managing expensive objects."""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 10, 
                 min_size: int = 1, timeout_seconds: float = 30.0):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.timeout_seconds = timeout_seconds
        self.logger = StructuredLogger("resource_pool")
        
        self._pool: queue.Queue[T] = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.Lock()
        self._stats = {
            "borrowed": 0,
            "returned": 0,
            "created": 0,
            "destroyed": 0,
            "timeouts": 0
        }
        
        # Pre-populate with minimum resources
        self._populate_minimum()
    
    def _populate_minimum(self) -> None:
        """Create minimum number of resources."""
        with self._lock:
            for _ in range(self.min_size):
                if self._created_count < self.max_size:
                    resource = self.factory()
                    self._pool.put(resource)
                    self._created_count += 1
                    self._stats["created"] += 1
    
    def borrow(self) -> Optional[T]:
        """Borrow a resource from the pool."""
        try:
            # Try to get existing resource
            resource = self._pool.get(timeout=self.timeout_seconds)
            self._stats["borrowed"] += 1
            return resource
            
        except queue.Empty:
            # Try to create new resource if under limit
            with self._lock:
                if self._created_count < self.max_size:
                    try:
                        resource = self.factory()
                        self._created_count += 1
                        self._stats["created"] += 1
                        self._stats["borrowed"] += 1
                        return resource
                    except Exception as e:
                        self.logger.error("Failed to create resource", error=str(e))
                        return None
                else:
                    self._stats["timeouts"] += 1
                    self.logger.warning("Resource pool timeout", 
                                      max_size=self.max_size,
                                      timeout_seconds=self.timeout_seconds)
                    return None
    
    def return_resource(self, resource: T) -> None:
        """Return a resource to the pool."""
        try:
            self._pool.put(resource, block=False)
            self._stats["returned"] += 1
        except queue.Full:
            # Pool is full, destroy the resource
            with self._lock:
                self._created_count -= 1
                self._stats["destroyed"] += 1
    
    @contextmanager
    def get_resource(self):
        """Context manager for borrowing and returning resources."""
        resource = self.borrow()
        if resource is None:
            raise RuntimeError("Failed to acquire resource from pool")
        
        try:
            yield resource
        finally:
            self.return_resource(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "min_size": self.min_size,
                "current_size": self._pool.qsize(),
                "created_count": self._created_count,
                "borrowed": self._stats["borrowed"],
                "returned": self._stats["returned"],
                "created": self._stats["created"],
                "destroyed": self._stats["destroyed"],
                "timeouts": self._stats["timeouts"],
                "utilization": (self._stats["borrowed"] - self._stats["returned"]) / self.max_size,
            }
    
    def shutdown(self) -> None:
        """Shutdown the resource pool."""
        with self._lock:
            # Drain the pool
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                    self._stats["destroyed"] += 1
                except queue.Empty:
                    break
            self._created_count = 0


class CompilationWorker:
    """Worker process for handling compilation tasks."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.logger = StructuredLogger(f"worker_{worker_id}")
        self.processed_tasks = 0
        self.start_time = time.time()
    
    def process_network_compilation(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process a network compilation task."""
        task_id = task_data.get("task_id", "unknown")
        start_time = time.time()
        
        try:
            from ..network_compiler import NetworkCompiler
            from ..core import FPGATarget
            from ..models.optimization import OptimizationLevel
            
            # Extract task parameters
            network_path = Path(task_data["network_path"])
            target = FPGATarget(task_data["target"])
            output_dir = Path(task_data["output_dir"])
            optimization_level = OptimizationLevel(task_data.get("optimization_level", 1))
            
            # Create compiler
            compiler = NetworkCompiler(target, enable_monitoring=False)
            
            # Compile network
            result = compiler.compile(
                network_path,
                output_dir,
                config=type('Config', (), {
                    'optimization_level': optimization_level,
                    'clock_frequency': task_data.get("clock_frequency", 100_000_000),
                    'debug_enabled': task_data.get("debug_enabled", False),
                    'generate_reports': task_data.get("generate_reports", True),
                    'run_synthesis': task_data.get("run_synthesis", False)
                })()
            )
            
            self.processed_tasks += 1
            duration = time.time() - start_time
            
            if result.success:
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result={
                        "hdl_files": {k: str(v) for k, v in result.hdl_files.items()},
                        "resource_estimate": {
                            "neurons": result.resource_estimate.neurons,
                            "synapses": result.resource_estimate.synapses,
                            "luts": result.resource_estimate.luts,
                            "memory_kb": result.resource_estimate.bram_kb,
                            "dsp_slices": result.resource_estimate.dsp_slices,
                        },
                        "optimization_stats": result.optimization_stats,
                    },
                    duration_seconds=duration,
                    worker_id=self.worker_id
                )
            else:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error="; ".join(result.errors),
                    duration_seconds=duration,
                    worker_id=self.worker_id
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Task processing failed", 
                            task_id=task_id, 
                            error=str(e),
                            duration=duration)
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
                worker_id=self.worker_id
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.start_time
        return {
            "worker_id": self.worker_id,
            "processed_tasks": self.processed_tasks,
            "uptime_seconds": uptime,
            "tasks_per_second": self.processed_tasks / uptime if uptime > 0 else 0,
        }


class ConcurrentCompiler:
    """Manages concurrent compilation across multiple workers."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_processes = use_processes
        self.logger = StructuredLogger("concurrent_compiler")
        
        # Task management
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_counter = 0
        self.task_lock = threading.Lock()
        
        # Worker pool
        if use_processes:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_duration": 0.0,
        }
        
        self.logger.info("ConcurrentCompiler initialized",
                        max_workers=self.max_workers,
                        use_processes=use_processes)
    
    def submit_compilation(self, network_path: Path, target: str, output_dir: Path,
                          **kwargs) -> str:
        """Submit a compilation task for concurrent execution."""
        with self.task_lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
        
        task_data = {
            "task_id": task_id,
            "network_path": str(network_path),
            "target": target,
            "output_dir": str(output_dir),
            **kwargs
        }
        
        # Submit to executor
        future = self.executor.submit(self._process_task_wrapper, task_data)
        
        self.pending_tasks[task_id] = {
            "future": future,
            "submitted_at": time.time(),
            "task_data": task_data
        }
        
        self.stats["tasks_submitted"] += 1
        
        self.logger.info("Task submitted", task_id=task_id, target=target)
        return task_id
    
    def _process_task_wrapper(self, task_data: Dict[str, Any]) -> TaskResult:
        """Wrapper for task processing to handle worker creation."""
        worker_id = f"worker_{threading.current_thread().ident or multiprocessing.current_process().pid}"
        worker = CompilationWorker(worker_id)
        return worker.process_network_compilation(task_data)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "status": "completed",
                "success": result.success,
                "duration_seconds": result.duration_seconds,
                "worker_id": result.worker_id,
                "error": result.error if not result.success else None
            }
        
        if task_id in self.pending_tasks:
            task_info = self.pending_tasks[task_id]
            future = task_info["future"]
            
            if future.done():
                # Task completed, move to completed tasks
                try:
                    result = future.result()
                    self.completed_tasks[task_id] = result
                    del self.pending_tasks[task_id]
                    
                    # Update stats
                    if result.success:
                        self.stats["tasks_completed"] += 1
                    else:
                        self.stats["tasks_failed"] += 1
                    self.stats["total_duration"] += result.duration_seconds
                    
                    return {
                        "status": "completed",
                        "success": result.success,
                        "duration_seconds": result.duration_seconds,
                        "worker_id": result.worker_id,
                        "error": result.error if not result.success else None
                    }
                    
                except Exception as e:
                    # Task failed with exception
                    result = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        duration_seconds=time.time() - task_info["submitted_at"]
                    )
                    self.completed_tasks[task_id] = result
                    del self.pending_tasks[task_id]
                    self.stats["tasks_failed"] += 1
                    
                    return {
                        "status": "completed",
                        "success": False,
                        "error": str(e)
                    }
            else:
                return {
                    "status": "running",
                    "elapsed_seconds": time.time() - task_info["submitted_at"]
                }
        
        return {"status": "not_found"}
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Wait for a specific task to complete."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id not in self.pending_tasks:
            return None
        
        task_info = self.pending_tasks[task_id]
        future = task_info["future"]
        
        try:
            result = future.result(timeout=timeout)
            self.completed_tasks[task_id] = result
            del self.pending_tasks[task_id]
            
            # Update stats
            if result.success:
                self.stats["tasks_completed"] += 1
            else:
                self.stats["tasks_failed"] += 1
            self.stats["total_duration"] += result.duration_seconds
            
            return result
            
        except concurrent.futures.TimeoutError:
            return None
        except Exception as e:
            result = TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - task_info["submitted_at"]
            )
            self.completed_tasks[task_id] = result
            del self.pending_tasks[task_id]
            self.stats["tasks_failed"] += 1
            
            return result
    
    def wait_for_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for all pending tasks to complete."""
        results = []
        
        # Get all pending task IDs
        pending_ids = list(self.pending_tasks.keys())
        
        for task_id in pending_ids:
            result = self.wait_for_task(task_id, timeout)
            if result:
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        pending_count = len(self.pending_tasks)
        completed_count = len(self.completed_tasks)
        
        avg_duration = (self.stats["total_duration"] / 
                       (self.stats["tasks_completed"] + self.stats["tasks_failed"])
                       if (self.stats["tasks_completed"] + self.stats["tasks_failed"]) > 0 else 0)
        
        success_rate = (self.stats["tasks_completed"] / 
                       self.stats["tasks_submitted"] * 100
                       if self.stats["tasks_submitted"] > 0 else 0)
        
        return {
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "pending_tasks": pending_count,
            "completed_tasks": completed_count,
            "tasks_submitted": self.stats["tasks_submitted"],
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "success_rate_percent": success_rate,
            "average_duration_seconds": avg_duration,
            "total_duration_seconds": self.stats["total_duration"],
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the concurrent compiler."""
        self.logger.info("Shutting down concurrent compiler", 
                        pending_tasks=len(self.pending_tasks))
        
        if wait:
            # Wait for pending tasks
            self.wait_for_all(timeout=300)  # 5 minute timeout
        
        self.executor.shutdown(wait=wait)
        
        self.logger.info("Concurrent compiler shutdown complete",
                        final_stats=self.get_stats())


class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing compilation tasks."""
    
    def __init__(self, initial_workers: int = 4):
        self.logger = StructuredLogger("load_balancer")
        self.workers: List[ConcurrentCompiler] = []
        self.worker_stats: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Performance metrics
        self.task_history = []
        self.max_history = 100
        
        # Auto-scaling parameters
        self.min_workers = 1
        self.max_workers = min(multiprocessing.cpu_count() * 2, 16)
        self.scale_up_threshold = 0.8  # Scale up if utilization > 80%
        self.scale_down_threshold = 0.3  # Scale down if utilization < 30%
        self.scale_check_interval = 30  # Check every 30 seconds
        
        # Initialize workers
        for i in range(initial_workers):
            self._add_worker()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_and_scale)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Adaptive load balancer initialized",
                        initial_workers=initial_workers,
                        min_workers=self.min_workers,
                        max_workers=self.max_workers)
    
    def _add_worker(self) -> None:
        """Add a new worker."""
        with self.lock:
            worker_id = len(self.workers)
            worker = ConcurrentCompiler(max_workers=2, use_processes=True)
            self.workers.append(worker)
            self.worker_stats.append({"created_at": time.time(), "tasks_assigned": 0})
            
            self.logger.info("Added worker", worker_id=worker_id, total_workers=len(self.workers))
    
    def _remove_worker(self) -> None:
        """Remove a worker (gracefully)."""
        with self.lock:
            if len(self.workers) <= self.min_workers:
                return
            
            # Find worker with least load
            least_loaded_idx = min(
                range(len(self.workers)),
                key=lambda i: self.workers[i].get_stats()["pending_tasks"]
            )
            
            worker = self.workers.pop(least_loaded_idx)
            self.worker_stats.pop(least_loaded_idx)
            
            # Shutdown worker
            worker.shutdown(wait=False)
            
            self.logger.info("Removed worker", 
                           worker_id=least_loaded_idx, 
                           total_workers=len(self.workers))
    
    def _select_worker(self) -> ConcurrentCompiler:
        """Select the best worker for a new task."""
        with self.lock:
            if not self.workers:
                self._add_worker()
            
            # Select worker with lowest pending task count
            best_worker_idx = min(
                range(len(self.workers)),
                key=lambda i: self.workers[i].get_stats()["pending_tasks"]
            )
            
            self.worker_stats[best_worker_idx]["tasks_assigned"] += 1
            return self.workers[best_worker_idx]
    
    def submit_task(self, network_path: Path, target: str, output_dir: Path, **kwargs) -> str:
        """Submit a task to the best available worker."""
        worker = self._select_worker()
        task_id = worker.submit_compilation(network_path, target, output_dir, **kwargs)
        
        # Track task for performance monitoring
        self.task_history.append({
            "task_id": task_id,
            "submitted_at": time.time(),
            "worker": worker,
        })
        
        # Keep history bounded
        if len(self.task_history) > self.max_history:
            self.task_history = self.task_history[-self.max_history:]
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task across all workers."""
        for worker in self.workers:
            status = worker.get_task_status(task_id)
            if status["status"] != "not_found":
                return status
        
        return {"status": "not_found"}
    
    def _calculate_system_utilization(self) -> float:
        """Calculate overall system utilization."""
        if not self.workers:
            return 0.0
        
        total_capacity = sum(w.max_workers for w in self.workers)
        total_pending = sum(w.get_stats()["pending_tasks"] for w in self.workers)
        
        return total_pending / total_capacity if total_capacity > 0 else 0.0
    
    def _monitor_and_scale(self) -> None:
        """Monitor performance and auto-scale workers."""
        while self.monitoring:
            try:
                time.sleep(self.scale_check_interval)
                
                utilization = self._calculate_system_utilization()
                current_workers = len(self.workers)
                
                # Scale up if high utilization
                if (utilization > self.scale_up_threshold and 
                    current_workers < self.max_workers):
                    self._add_worker()
                    self.logger.info("Scaled up workers", 
                                   utilization=utilization,
                                   worker_count=len(self.workers))
                
                # Scale down if low utilization
                elif (utilization < self.scale_down_threshold and 
                      current_workers > self.min_workers):
                    self._remove_worker()
                    self.logger.info("Scaled down workers",
                                   utilization=utilization,
                                   worker_count=len(self.workers))
                
            except Exception as e:
                self.logger.error("Error in monitoring thread", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self.lock:
            worker_stats = [w.get_stats() for w in self.workers]
            
            total_pending = sum(stats["pending_tasks"] for stats in worker_stats)
            total_completed = sum(stats["tasks_completed"] for stats in worker_stats)
            total_failed = sum(stats["tasks_failed"] for stats in worker_stats)
            
            return {
                "worker_count": len(self.workers),
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "system_utilization": self._calculate_system_utilization(),
                "total_pending_tasks": total_pending,
                "total_completed_tasks": total_completed,
                "total_failed_tasks": total_failed,
                "worker_stats": worker_stats,
                "system_resources": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_count": psutil.cpu_count(),
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown the load balancer and all workers."""
        self.monitoring = False
        
        with self.lock:
            for i, worker in enumerate(self.workers):
                self.logger.info("Shutting down worker", worker_id=i)
                worker.shutdown(wait=False)
        
        self.logger.info("Load balancer shutdown complete")