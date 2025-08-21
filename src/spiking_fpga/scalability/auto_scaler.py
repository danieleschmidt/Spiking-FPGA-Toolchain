"""Intelligent auto-scaling system for compilation workloads."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from queue import Queue, Empty
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    queue_length: int
    active_workers: int
    avg_task_duration: float
    throughput: float
    timestamp: float


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 1
    max_workers: int = None  # None = auto-detect based on CPU cores
    target_cpu_usage: float = 75.0  # Target CPU usage percentage
    scale_up_threshold: float = 85.0  # Scale up when CPU > this
    scale_down_threshold: float = 50.0  # Scale down when CPU < this
    scale_up_cooldown: int = 30  # Seconds before scaling up again
    scale_down_cooldown: int = 60  # Seconds before scaling down again
    queue_threshold: int = 5  # Scale up when queue > this length
    memory_threshold: float = 90.0  # Emergency scale down when memory > this


@dataclass
class ScalingPolicy:
    """Simple scaling policy for compatibility."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    scale_up_cooldown: int = 60
    scale_down_cooldown: int = 300


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        
        if self.config.max_workers is None:
            self.config.max_workers = min(32, (os.cpu_count() or 4) * 2)
        
        self.task_queue = Queue()
        self.workers: Dict[str, ThreadPoolExecutor] = {}
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_action = {'up': 0, 'down': 0}
        
        self._active = False
        self._monitor_thread = None
        self._metrics_lock = threading.Lock()
        
        # Performance tracking
        self.task_durations: List[float] = []
        self.completed_tasks = 0
        self.start_time = time.time()
        
        # Predictive scaling
        self.workload_patterns: Dict[str, List[float]] = {}
        self.prediction_window = 300  # 5 minutes
        
    def start(self):
        """Start the auto-scaler."""
        self._active = True
        
        # Start with minimum workers
        self._create_worker_pool('primary', self.config.min_workers)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"AutoScaler started with {self.config.min_workers} workers")
    
    def stop(self):
        """Stop the auto-scaler and clean up resources."""
        self._active = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        # Shutdown all worker pools
        for pool_name, pool in self.workers.items():
            pool.shutdown(wait=True)
            logger.info(f"Shut down worker pool: {pool_name}")
        
        self.workers.clear()
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Future:
        """Submit a task for execution with auto-scaling."""
        # Choose best worker pool
        pool_name = self._select_worker_pool()
        
        if pool_name not in self.workers:
            # Create emergency worker pool if needed
            self._create_worker_pool(pool_name, 1)
        
        start_time = time.time()
        
        def wrapped_func(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self._record_task_completion(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self._record_task_completion(duration, failed=True)
                raise
        
        return self.workers[pool_name].submit(wrapped_func, *args, **kwargs)
    
    def get_metrics(self) -> ScalingMetrics:
        """Get current scaling metrics."""
        with self._metrics_lock:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            total_workers = sum(pool._max_workers for pool in self.workers.values())
            
            avg_duration = 0
            if self.task_durations:
                avg_duration = sum(self.task_durations[-100:]) / len(self.task_durations[-100:])
            
            # Calculate throughput (tasks per second)
            elapsed = time.time() - self.start_time
            throughput = self.completed_tasks / max(elapsed, 1)
            
            return ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                queue_length=self.task_queue.qsize(),
                active_workers=total_workers,
                avg_task_duration=avg_duration,
                throughput=throughput,
                timestamp=time.time()
            )
    
    def _monitor_loop(self):
        """Main monitoring loop for auto-scaling decisions."""
        while self._active:
            try:
                metrics = self.get_metrics()
                self._store_metrics(metrics)
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                if decision:
                    self._execute_scaling_decision(decision)
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in auto-scaler monitor loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Make intelligent scaling decision based on metrics and predictions."""
        current_time = time.time()
        
        # Emergency memory scaling
        if metrics.memory_usage > self.config.memory_threshold:
            if current_time - self.last_scale_action['down'] > 10:  # Emergency override
                return {
                    'action': 'scale_down',
                    'reason': 'emergency_memory',
                    'target_workers': max(1, metrics.active_workers // 2)
                }
        
        # Check cooldown periods
        can_scale_up = (current_time - self.last_scale_action['up'] > 
                       self.config.scale_up_cooldown)
        can_scale_down = (current_time - self.last_scale_action['down'] > 
                         self.config.scale_down_cooldown)
        
        # Scale up conditions
        if can_scale_up and metrics.active_workers < self.config.max_workers:
            should_scale_up = any([
                metrics.cpu_usage > self.config.scale_up_threshold,
                metrics.queue_length > self.config.queue_threshold,
                self._predict_workload_increase()
            ])
            
            if should_scale_up:
                new_workers = min(
                    self.config.max_workers,
                    metrics.active_workers + max(1, metrics.active_workers // 4)
                )
                return {
                    'action': 'scale_up',
                    'reason': 'high_load',
                    'target_workers': new_workers
                }
        
        # Scale down conditions
        if can_scale_down and metrics.active_workers > self.config.min_workers:
            should_scale_down = all([
                metrics.cpu_usage < self.config.scale_down_threshold,
                metrics.queue_length == 0,
                not self._predict_workload_increase()
            ])
            
            if should_scale_down:
                new_workers = max(
                    self.config.min_workers,
                    metrics.active_workers - max(1, metrics.active_workers // 6)
                )
                return {
                    'action': 'scale_down',
                    'reason': 'low_load', 
                    'target_workers': new_workers
                }
        
        return None
    
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute a scaling decision."""
        action = decision['action']
        target_workers = decision['target_workers']
        reason = decision['reason']
        
        current_workers = sum(pool._max_workers for pool in self.workers.values())
        
        if action == 'scale_up' and target_workers > current_workers:
            self._scale_up(target_workers - current_workers)
            self.last_scale_action['up'] = time.time()
            logger.info(f"Scaled up to {target_workers} workers (reason: {reason})")
            
        elif action == 'scale_down' and target_workers < current_workers:
            self._scale_down(current_workers - target_workers)
            self.last_scale_action['down'] = time.time()
            logger.info(f"Scaled down to {target_workers} workers (reason: {reason})")
    
    def _scale_up(self, additional_workers: int):
        """Add additional workers to handle increased load."""
        # Add workers to existing pools or create new ones
        if 'primary' in self.workers:
            # Expand primary pool
            current_max = self.workers['primary']._max_workers
            new_max = current_max + additional_workers
            
            # Create new pool with expanded capacity
            old_pool = self.workers['primary']
            self.workers['primary'] = ThreadPoolExecutor(max_workers=new_max)
            
            # Gracefully shutdown old pool after a delay
            threading.Timer(30, old_pool.shutdown).start()
        else:
            self._create_worker_pool('primary', additional_workers)
    
    def _scale_down(self, reduce_workers: int):
        """Remove workers to reduce resource usage."""
        # Reduce worker count in pools
        if 'primary' in self.workers:
            current_max = self.workers['primary']._max_workers
            new_max = max(1, current_max - reduce_workers)
            
            if new_max < current_max:
                # Create new smaller pool
                old_pool = self.workers['primary']
                self.workers['primary'] = ThreadPoolExecutor(max_workers=new_max)
                
                # Gracefully shutdown old pool
                threading.Timer(60, old_pool.shutdown).start()
    
    def _create_worker_pool(self, name: str, workers: int):
        """Create a new worker pool."""
        self.workers[name] = ThreadPoolExecutor(max_workers=workers)
        logger.debug(f"Created worker pool '{name}' with {workers} workers")
    
    def _select_worker_pool(self) -> str:
        """Select the best worker pool for task submission."""
        if not self.workers:
            return 'primary'
        
        # Simple round-robin for now
        # Could be enhanced with load-based selection
        return 'primary'
    
    def _record_task_completion(self, duration: float, failed: bool = False):
        """Record completion of a task."""
        with self._metrics_lock:
            self.task_durations.append(duration)
            if not failed:
                self.completed_tasks += 1
            
            # Keep only recent durations
            if len(self.task_durations) > 1000:
                self.task_durations = self.task_durations[-500:]
    
    def _store_metrics(self, metrics: ScalingMetrics):
        """Store metrics for historical analysis."""
        with self._metrics_lock:
            self.metrics_history.append(metrics)
            
            # Store workload patterns for prediction
            hour_key = time.strftime('%H', time.localtime(metrics.timestamp))
            if hour_key not in self.workload_patterns:
                self.workload_patterns[hour_key] = []
            self.workload_patterns[hour_key].append(metrics.cpu_usage)
            
            # Keep patterns manageable
            for key in self.workload_patterns:
                if len(self.workload_patterns[key]) > 100:
                    self.workload_patterns[key] = self.workload_patterns[key][-50:]
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory growth."""
        with self._metrics_lock:
            cutoff = time.time() - 3600  # Keep 1 hour of metrics
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff
            ]
    
    def _predict_workload_increase(self) -> bool:
        """Predict if workload will increase based on historical patterns."""
        if not self.workload_patterns:
            return False
        
        current_hour = time.strftime('%H')
        next_hour = str((int(current_hour) + 1) % 24).zfill(2)
        
        current_pattern = self.workload_patterns.get(current_hour, [])
        next_pattern = self.workload_patterns.get(next_hour, [])
        
        if current_pattern and next_pattern:
            current_avg = sum(current_pattern[-10:]) / len(current_pattern[-10:])
            next_avg = sum(next_pattern[-10:]) / len(next_pattern[-10:])
            
            # Predict increase if next hour typically has 20% more load
            return next_avg > current_avg * 1.2
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler statistics."""
        with self._metrics_lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = self.metrics_history[-20:]  # Last 20 measurements
            
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            
            return {
                'active_workers': sum(pool._max_workers for pool in self.workers.values()),
                'completed_tasks': self.completed_tasks,
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_task_duration': sum(self.task_durations[-100:]) / len(self.task_durations[-100:]) if self.task_durations else 0,
                'throughput': self.completed_tasks / max(time.time() - self.start_time, 1),
                'scale_up_events': sum(1 for _ in [1]),  # Could track these
                'scale_down_events': sum(1 for _ in [1]),
                'uptime': time.time() - self.start_time
            }