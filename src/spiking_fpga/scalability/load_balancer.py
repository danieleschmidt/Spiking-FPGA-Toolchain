"""Intelligent load balancer for distributed compilation workloads."""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue, PriorityQueue, Empty
import logging
import hashlib
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    worker_id: str
    capacity: int
    current_load: int
    performance_score: float
    last_heartbeat: float
    specializations: List[str]
    status: str = 'active'  # active, busy, offline
    
    @property
    def utilization(self) -> float:
        """Get current utilization percentage."""
        return (self.current_load / max(self.capacity, 1)) * 100
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return self.status == 'active' and self.current_load < self.capacity


@dataclass
class Task:
    """Represents a task to be load balanced."""
    task_id: str
    priority: int
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    specialization_required: Optional[str] = None
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)."""
        return self.priority > other.priority


class LoadBalancer:
    """Intelligent load balancer with adaptive algorithms."""
    
    def __init__(self, strategy: str = 'adaptive'):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = PriorityQueue()
        self.assignment_history: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.load_history: List[float] = []
        
        # Configuration
        self.heartbeat_timeout = 30  # seconds
        self.rebalance_interval = 60  # seconds
        self.performance_window = 100  # number of recent tasks to consider
        
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        
    def start(self):
        """Start the load balancer monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Load balancer started")
    
    def stop(self):
        """Stop the load balancer."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Load balancer stopped")
    
    def register_worker(self, worker_id: str, capacity: int, 
                       specializations: Optional[List[str]] = None):
        """Register a new worker node."""
        with self._lock:
            worker = WorkerNode(
                worker_id=worker_id,
                capacity=capacity,
                current_load=0,
                performance_score=1.0,
                last_heartbeat=time.time(),
                specializations=specializations or []
            )
            self.workers[worker_id] = worker
        
        logger.info(f"Registered worker {worker_id} with capacity {capacity}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Unregistered worker {worker_id}")
    
    def update_worker_status(self, worker_id: str, status: str, 
                           current_load: Optional[int] = None):
        """Update worker status and load."""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.status = status
                worker.last_heartbeat = time.time()
                
                if current_load is not None:
                    worker.current_load = current_load
    
    def submit_task(self, task: Task) -> Optional[str]:
        """Submit a task and get assigned worker ID."""
        with self._lock:
            # Find best worker for this task
            selected_worker = self._select_worker(task)
            
            if selected_worker:
                # Assign task to worker
                self.workers[selected_worker].current_load += 1
                self.assignment_history[selected_worker].append(task.task_id)
                
                # Keep assignment history manageable
                if len(self.assignment_history[selected_worker]) > 1000:
                    self.assignment_history[selected_worker] = \
                        self.assignment_history[selected_worker][-500:]
                
                logger.debug(f"Assigned task {task.task_id} to worker {selected_worker}")
                return selected_worker
            else:
                # No available workers, queue the task
                self.task_queue.put(task)
                logger.debug(f"Queued task {task.task_id} - no available workers")
                return None
    
    def complete_task(self, worker_id: str, task_id: str, 
                     duration: float, success: bool):
        """Mark task as completed and update worker metrics."""
        with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_load = max(0, worker.current_load - 1)
                
                # Update performance metrics
                if success:
                    self.performance_metrics[worker_id].append(duration)
                    
                    # Keep only recent performance data
                    if len(self.performance_metrics[worker_id]) > self.performance_window:
                        self.performance_metrics[worker_id] = \
                            self.performance_metrics[worker_id][-50:]
                    
                    # Update performance score
                    self._update_performance_score(worker_id)
                
                # Try to assign queued tasks
                self._assign_queued_tasks()
    
    def get_worker_recommendations(self, task: Task) -> List[str]:
        """Get ranked list of worker recommendations for a task."""
        with self._lock:
            available_workers = [
                (worker_id, worker) for worker_id, worker in self.workers.items()
                if worker.is_available
            ]
            
            if not available_workers:
                return []
            
            # Score workers based on various factors
            scored_workers = []
            for worker_id, worker in available_workers:
                score = self._calculate_worker_score(worker, task)
                scored_workers.append((score, worker_id))
            
            # Sort by score (highest first)
            scored_workers.sort(reverse=True)
            
            return [worker_id for _, worker_id in scored_workers]
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics."""
        with self._lock:
            total_capacity = sum(w.capacity for w in self.workers.values())
            total_load = sum(w.current_load for w in self.workers.values())
            
            active_workers = sum(1 for w in self.workers.values() if w.is_available)
            
            # Calculate average performance
            all_performances = []
            for worker_performances in self.performance_metrics.values():
                all_performances.extend(worker_performances)
            
            avg_performance = sum(all_performances) / len(all_performances) if all_performances else 0
            
            return {
                'total_workers': len(self.workers),
                'active_workers': active_workers,
                'total_capacity': total_capacity,
                'current_load': total_load,
                'utilization_percent': (total_load / max(total_capacity, 1)) * 100,
                'queued_tasks': self.task_queue.qsize(),
                'avg_task_duration': avg_performance,
                'worker_details': {
                    worker_id: {
                        'capacity': worker.capacity,
                        'current_load': worker.current_load,
                        'utilization': worker.utilization,
                        'performance_score': worker.performance_score,
                        'status': worker.status
                    }
                    for worker_id, worker in self.workers.items()
                }
            }
    
    def rebalance_load(self) -> Dict[str, List[str]]:
        """Perform load rebalancing and return recommended task migrations."""
        with self._lock:
            if len(self.workers) < 2:
                return {}
            
            # Find overloaded and underloaded workers
            overloaded = []
            underloaded = []
            
            avg_utilization = sum(w.utilization for w in self.workers.values()) / len(self.workers)
            
            for worker_id, worker in self.workers.items():
                if worker.utilization > avg_utilization * 1.3:  # 30% above average
                    overloaded.append((worker_id, worker))
                elif worker.utilization < avg_utilization * 0.7:  # 30% below average
                    underloaded.append((worker_id, worker))
            
            # Generate migration recommendations
            migrations = {}
            
            for overloaded_id, overloaded_worker in overloaded:
                if not underloaded:
                    break
                
                # Find best target for migration
                target_id, target_worker = min(underloaded, 
                                             key=lambda x: x[1].utilization)
                
                # Recommend migrating some tasks
                tasks_to_migrate = min(2, overloaded_worker.current_load // 4)
                
                if tasks_to_migrate > 0:
                    migrations[overloaded_id] = [target_id] * tasks_to_migrate
                    
                    # Update temporary utilization for next iteration
                    overloaded_worker.current_load -= tasks_to_migrate
                    target_worker.current_load += tasks_to_migrate
            
            return migrations
    
    def _select_worker(self, task: Task) -> Optional[str]:
        """Select the best worker for a task based on current strategy."""
        available_workers = [
            (worker_id, worker) for worker_id, worker in self.workers.items()
            if worker.is_available
        ]
        
        if not available_workers:
            return None
        
        if self.strategy == 'round_robin':
            return self._round_robin_selection(available_workers)
        elif self.strategy == 'least_loaded':
            return self._least_loaded_selection(available_workers)
        elif self.strategy == 'performance_based':
            return self._performance_based_selection(available_workers, task)
        elif self.strategy == 'adaptive':
            return self._adaptive_selection(available_workers, task)
        else:
            # Default to least loaded
            return self._least_loaded_selection(available_workers)
    
    def _round_robin_selection(self, workers: List[tuple]) -> str:
        """Round-robin worker selection."""
        # Simple hash-based round robin
        worker_ids = [worker_id for worker_id, _ in workers]
        index = int(time.time()) % len(worker_ids)
        return worker_ids[index]
    
    def _least_loaded_selection(self, workers: List[tuple]) -> str:
        """Select worker with lowest current load."""
        return min(workers, key=lambda x: x[1].utilization)[0]
    
    def _performance_based_selection(self, workers: List[tuple], task: Task) -> str:
        """Select worker based on historical performance."""
        best_worker = None
        best_score = -1
        
        for worker_id, worker in workers:
            score = worker.performance_score
            
            # Prefer workers with relevant specializations
            if (task.specialization_required and 
                task.specialization_required in worker.specializations):
                score *= 1.5
            
            # Penalty for high utilization
            score *= (1 - worker.utilization / 100)
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        return best_worker or workers[0][0]
    
    def _adaptive_selection(self, workers: List[tuple], task: Task) -> str:
        """Adaptive selection combining multiple factors."""
        scored_workers = []
        
        for worker_id, worker in workers:
            score = self._calculate_worker_score(worker, task)
            scored_workers.append((score, worker_id))
        
        # Add some randomness to prevent all tasks going to one worker
        scored_workers.sort(reverse=True)
        
        # Weighted random selection from top 3 workers
        top_workers = scored_workers[:min(3, len(scored_workers))]
        weights = [score for score, _ in top_workers]
        
        if weights:
            total_weight = sum(weights)
            r = random.uniform(0, total_weight)
            
            cumulative = 0
            for (score, worker_id) in top_workers:
                cumulative += score
                if r <= cumulative:
                    return worker_id
        
        return scored_workers[0][1] if scored_workers else workers[0][0]
    
    def _calculate_worker_score(self, worker: WorkerNode, task: Task) -> float:
        """Calculate a comprehensive score for worker-task matching."""
        score = 1.0
        
        # Performance factor (higher performance = higher score)
        score *= worker.performance_score
        
        # Load factor (lower utilization = higher score)
        load_factor = 1 - (worker.utilization / 100)
        score *= max(0.1, load_factor)  # Minimum score of 0.1
        
        # Specialization bonus
        if (task.specialization_required and 
            task.specialization_required in worker.specializations):
            score *= 2.0
        
        # Capacity factor
        capacity_factor = worker.capacity / max(1, worker.current_load + 1)
        score *= min(2.0, capacity_factor)  # Cap at 2x bonus
        
        # Recent assignment diversity (prefer workers with fewer recent assignments)
        recent_assignments = len(self.assignment_history[worker.worker_id][-10:])
        diversity_factor = 1 / max(1, recent_assignments * 0.1)
        score *= diversity_factor
        
        return score
    
    def _update_performance_score(self, worker_id: str):
        """Update performance score for a worker based on recent tasks."""
        if worker_id not in self.performance_metrics:
            return
        
        recent_durations = self.performance_metrics[worker_id][-20:]  # Last 20 tasks
        
        if len(recent_durations) < 3:
            return  # Need more data
        
        # Calculate performance relative to average
        avg_duration = sum(recent_durations) / len(recent_durations)
        
        # Compare to global average
        all_durations = []
        for durations in self.performance_metrics.values():
            all_durations.extend(durations[-20:])
        
        if all_durations:
            global_avg = sum(all_durations) / len(all_durations)
            
            # Performance score: faster = higher score
            if avg_duration > 0:
                performance_ratio = global_avg / avg_duration
                # Smooth the update
                current_score = self.workers[worker_id].performance_score
                new_score = (current_score * 0.7) + (performance_ratio * 0.3)
                self.workers[worker_id].performance_score = max(0.1, min(3.0, new_score))
    
    def _assign_queued_tasks(self):
        """Try to assign queued tasks to available workers."""
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                selected_worker = self._select_worker(task)
                
                if selected_worker:
                    self.workers[selected_worker].current_load += 1
                    self.assignment_history[selected_worker].append(task.task_id)
                    logger.debug(f"Assigned queued task {task.task_id} to worker {selected_worker}")
                else:
                    # Put back in queue
                    self.task_queue.put(task)
                    break
                    
            except Empty:
                break
    
    def _monitor_loop(self):
        """Monitor worker health and perform periodic maintenance."""
        last_rebalance = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Check worker heartbeats
                with self._lock:
                    offline_workers = []
                    for worker_id, worker in self.workers.items():
                        if (current_time - worker.last_heartbeat) > self.heartbeat_timeout:
                            worker.status = 'offline'
                            offline_workers.append(worker_id)
                    
                    if offline_workers:
                        logger.warning(f"Workers offline: {offline_workers}")
                
                # Periodic rebalancing
                if current_time - last_rebalance > self.rebalance_interval:
                    migrations = self.rebalance_load()
                    if migrations:
                        logger.info(f"Load rebalancing recommended: {migrations}")
                    last_rebalance = current_time
                
                # Update load history
                stats = self.get_load_statistics()
                self.load_history.append(stats['utilization_percent'])
                
                # Keep history manageable
                if len(self.load_history) > 1000:
                    self.load_history = self.load_history[-500:]
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in load balancer monitor loop: {e}")
                time.sleep(30)  # Wait longer on error