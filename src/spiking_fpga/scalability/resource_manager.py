"""Advanced resource management for optimal FPGA utilization."""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import psutil
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class ResourceQuota:
    """Resource quota for a compilation session."""
    max_memory_mb: int
    max_cpu_percent: float
    max_disk_gb: float
    max_duration_seconds: int
    priority: int = 1


@dataclass
class ResourceUsage:
    """Current resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    disk_gb: float
    duration_seconds: float
    network_io_mb: float = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class FPGAResource:
    """FPGA resource allocation tracking."""
    target: str
    logic_cells_used: int
    logic_cells_total: int
    bram_kb_used: float
    bram_kb_total: float
    dsp_blocks_used: int
    dsp_blocks_total: int
    
    @property
    def logic_utilization(self) -> float:
        return (self.logic_cells_used / max(1, self.logic_cells_total)) * 100
    
    @property
    def memory_utilization(self) -> float:
        return (self.bram_kb_used / max(1, self.bram_kb_total)) * 100
    
    @property
    def dsp_utilization(self) -> float:
        return (self.dsp_blocks_used / max(1, self.dsp_blocks_total)) * 100
    
    @property
    def overall_utilization(self) -> float:
        return max(self.logic_utilization, self.memory_utilization, self.dsp_utilization)


class ResourceManager:
    """Advanced resource management with intelligent allocation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Resource tracking
        self.active_sessions: Dict[str, ResourceQuota] = {}
        self.usage_history: Dict[str, List[ResourceUsage]] = defaultdict(list)
        self.fpga_allocations: Dict[str, FPGAResource] = {}
        
        # System limits
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.system_cpu_cores = psutil.cpu_count()
        self.system_disk_gb = psutil.disk_usage('/').total / (1024**3)
        
        # Resource pools
        self.memory_pool = self.config.get('memory_pool_gb', self.system_memory_gb * 0.8)
        self.cpu_pool = self.config.get('cpu_pool_percent', 80.0)
        self.disk_pool = self.config.get('disk_pool_gb', 100.0)
        
        # Allocation tracking
        self.allocated_memory = 0
        self.allocated_cpu = 0
        self.allocated_disk = 0
        
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance optimization
        self.resource_predictions: Dict[str, float] = {}
        self.optimization_cache: Dict[str, Dict[str, Any]] = {}
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource manager monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource manager monitoring stopped")
    
    def request_resources(self, session_id: str, 
                         quota: ResourceQuota) -> Tuple[bool, Optional[str]]:
        """Request resource allocation for a compilation session."""
        with self._lock:
            # Check if resources are available
            if not self._can_allocate(quota):
                reason = self._get_allocation_failure_reason(quota)
                return False, reason
            
            # Allocate resources
            self.active_sessions[session_id] = quota
            self.allocated_memory += quota.max_memory_mb / 1024  # Convert to GB
            self.allocated_cpu += quota.max_cpu_percent
            self.allocated_disk += quota.max_disk_gb
            
            logger.info(f"Allocated resources for session {session_id}")
            return True, None
    
    def release_resources(self, session_id: str):
        """Release resources for a completed session."""
        with self._lock:
            if session_id in self.active_sessions:
                quota = self.active_sessions[session_id]
                
                # Deallocate
                self.allocated_memory -= quota.max_memory_mb / 1024
                self.allocated_cpu -= quota.max_cpu_percent
                self.allocated_disk -= quota.max_disk_gb
                
                # Ensure non-negative
                self.allocated_memory = max(0, self.allocated_memory)
                self.allocated_cpu = max(0, self.allocated_cpu)
                self.allocated_disk = max(0, self.allocated_disk)
                
                del self.active_sessions[session_id]
                logger.info(f"Released resources for session {session_id}")
    
    def record_usage(self, session_id: str, usage: ResourceUsage):
        """Record resource usage for a session."""
        with self._lock:
            self.usage_history[session_id].append(usage)
            
            # Keep history manageable
            if len(self.usage_history[session_id]) > 1000:
                self.usage_history[session_id] = self.usage_history[session_id][-500:]
    
    def allocate_fpga_resources(self, session_id: str, 
                               target: str, 
                               estimated_resources: Dict[str, int]) -> bool:
        """Allocate FPGA resources for a compilation."""
        with self._lock:
            # Get target FPGA specifications
            from spiking_fpga.core import FPGATarget
            
            try:
                fpga_target = FPGATarget(target)
                target_resources = fpga_target.resources
            except (ValueError, AttributeError):
                logger.error(f"Unknown FPGA target: {target}")
                return False
            
            # Check if allocation is possible
            required_logic = estimated_resources.get('logic_cells', 0)
            required_bram = estimated_resources.get('bram_kb', 0)
            required_dsp = estimated_resources.get('dsp_blocks', 0)
            
            max_logic = target_resources.get('logic_cells', 0)
            max_bram = target_resources.get('bram_kb', 0)
            max_dsp = target_resources.get('dsp_slices', 0)
            
            if (required_logic > max_logic or 
                required_bram > max_bram or 
                required_dsp > max_dsp):
                logger.warning(f"FPGA resource requirements exceed target capacity")
                return False
            
            # Allocate FPGA resources
            fpga_resource = FPGAResource(
                target=target,
                logic_cells_used=required_logic,
                logic_cells_total=max_logic,
                bram_kb_used=required_bram,
                bram_kb_total=max_bram,
                dsp_blocks_used=required_dsp,
                dsp_blocks_total=max_dsp
            )
            
            self.fpga_allocations[session_id] = fpga_resource
            logger.info(f"Allocated FPGA resources for {session_id}: "
                       f"{fpga_resource.overall_utilization:.1f}% utilization")
            return True
    
    def release_fpga_resources(self, session_id: str):
        """Release FPGA resources."""
        with self._lock:
            if session_id in self.fpga_allocations:
                del self.fpga_allocations[session_id]
                logger.info(f"Released FPGA resources for session {session_id}")
    
    def optimize_resource_allocation(self, network_config: Dict[str, Any], 
                                   target: str) -> ResourceQuota:
        """Optimize resource allocation based on network characteristics."""
        # Generate cache key
        config_hash = str(hash(str(sorted(network_config.items()))))
        cache_key = f"{config_hash}_{target}"
        
        if cache_key in self.optimization_cache:
            cached = self.optimization_cache[cache_key]
            logger.debug(f"Using cached resource optimization for {cache_key}")
            return ResourceQuota(**cached)
        
        # Analyze network complexity
        complexity_score = self._calculate_network_complexity(network_config)
        
        # Base resource requirements
        base_memory = 512  # MB
        base_cpu = 25.0    # %
        base_disk = 2.0    # GB
        base_duration = 120  # seconds
        
        # Scale based on complexity
        memory_mb = int(base_memory * (1 + complexity_score))
        cpu_percent = min(90.0, base_cpu * (1 + complexity_score * 0.5))
        disk_gb = base_disk * (1 + complexity_score * 0.3)
        duration_seconds = int(base_duration * (1 + complexity_score))
        
        # Target-specific adjustments
        target_multipliers = {
            'artix7_35t': {'memory': 1.0, 'cpu': 1.0, 'time': 1.0},
            'artix7_100t': {'memory': 1.2, 'cpu': 1.1, 'time': 1.2},
            'cyclone5_gx': {'memory': 1.1, 'cpu': 1.05, 'time': 1.1},
            'cyclone5_gt': {'memory': 1.3, 'cpu': 1.15, 'time': 1.3}
        }
        
        multiplier = target_multipliers.get(target, 
                                          {'memory': 1.0, 'cpu': 1.0, 'time': 1.0})
        
        memory_mb = int(memory_mb * multiplier['memory'])
        cpu_percent *= multiplier['cpu']
        duration_seconds = int(duration_seconds * multiplier['time'])
        
        # Historical optimization
        if target in self.resource_predictions:
            prediction_factor = self.resource_predictions[target]
            memory_mb = int(memory_mb * prediction_factor)
            duration_seconds = int(duration_seconds * prediction_factor)
        
        quota = ResourceQuota(
            max_memory_mb=memory_mb,
            max_cpu_percent=cpu_percent,
            max_disk_gb=disk_gb,
            max_duration_seconds=duration_seconds,
            priority=1
        )
        
        # Cache the optimization
        self.optimization_cache[cache_key] = {
            'max_memory_mb': memory_mb,
            'max_cpu_percent': cpu_percent,
            'max_disk_gb': disk_gb,
            'max_duration_seconds': duration_seconds,
            'priority': 1
        }
        
        # Keep cache manageable
        if len(self.optimization_cache) > 1000:
            # Remove oldest entries
            items = list(self.optimization_cache.items())
            self.optimization_cache = dict(items[-500:])
        
        logger.info(f"Optimized resource allocation: {memory_mb}MB, "
                   f"{cpu_percent:.1f}% CPU, {duration_seconds}s")
        
        return quota
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        with self._lock:
            # Current system usage
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=1)
            system_disk = psutil.disk_usage('/')
            
            # Pool utilization
            memory_pool_used = (self.allocated_memory / max(1, self.memory_pool)) * 100
            cpu_pool_used = (self.allocated_cpu / max(1, self.cpu_pool)) * 100
            disk_pool_used = (self.allocated_disk / max(1, self.disk_pool)) * 100
            
            # Active sessions statistics
            active_sessions_count = len(self.active_sessions)
            total_allocated_memory = sum(s.max_memory_mb for s in self.active_sessions.values())
            total_allocated_cpu = sum(s.max_cpu_percent for s in self.active_sessions.values())
            
            # FPGA utilization
            fpga_stats = {}
            for session_id, fpga_alloc in self.fpga_allocations.items():
                if fpga_alloc.target not in fpga_stats:
                    fpga_stats[fpga_alloc.target] = {
                        'sessions': 0,
                        'avg_utilization': 0,
                        'peak_utilization': 0
                    }
                
                fpga_stats[fpga_alloc.target]['sessions'] += 1
                util = fpga_alloc.overall_utilization
                fpga_stats[fpga_alloc.target]['avg_utilization'] += util
                fpga_stats[fpga_alloc.target]['peak_utilization'] = max(
                    fpga_stats[fpga_alloc.target]['peak_utilization'], util)
            
            # Average utilization
            for target_stats in fpga_stats.values():
                if target_stats['sessions'] > 0:
                    target_stats['avg_utilization'] /= target_stats['sessions']
            
            return {
                'system_resources': {
                    'memory_total_gb': self.system_memory_gb,
                    'memory_used_percent': system_memory.percent,
                    'cpu_cores': self.system_cpu_cores,
                    'cpu_used_percent': system_cpu,
                    'disk_total_gb': system_disk.total / (1024**3),
                    'disk_used_percent': (system_disk.used / system_disk.total) * 100
                },
                'resource_pools': {
                    'memory_pool_gb': self.memory_pool,
                    'memory_pool_used_percent': memory_pool_used,
                    'cpu_pool_percent': self.cpu_pool,
                    'cpu_pool_used_percent': cpu_pool_used,
                    'disk_pool_gb': self.disk_pool,
                    'disk_pool_used_percent': disk_pool_used
                },
                'active_sessions': {
                    'count': active_sessions_count,
                    'total_memory_mb': total_allocated_memory,
                    'total_cpu_percent': total_allocated_cpu
                },
                'fpga_utilization': fpga_stats,
                'optimization_cache_size': len(self.optimization_cache)
            }
    
    def predict_resource_needs(self, network_configs: List[Dict[str, Any]], 
                              targets: List[str]) -> Dict[str, ResourceQuota]:
        """Predict resource needs for batch compilation."""
        predictions = {}
        
        for i, (config, target) in enumerate(zip(network_configs, targets)):
            session_id = f"batch_{i}"
            quota = self.optimize_resource_allocation(config, target)
            predictions[session_id] = quota
        
        return predictions
    
    def recommend_batch_scheduling(self, resource_predictions: Dict[str, ResourceQuota]) -> List[List[str]]:
        """Recommend optimal batching for resource-constrained compilation."""
        batches = []
        current_batch = []
        current_memory = 0
        current_cpu = 0
        current_disk = 0
        
        # Sort by resource requirements (largest first for better packing)
        sorted_sessions = sorted(
            resource_predictions.items(),
            key=lambda x: (x[1].max_memory_mb + x[1].max_cpu_percent * 10),
            reverse=True
        )
        
        for session_id, quota in sorted_sessions:
            # Check if adding this session would exceed limits
            memory_needed = current_memory + quota.max_memory_mb / 1024
            cpu_needed = current_cpu + quota.max_cpu_percent
            disk_needed = current_disk + quota.max_disk_gb
            
            if (memory_needed <= self.memory_pool * 0.9 and  # 90% utilization limit
                cpu_needed <= self.cpu_pool * 0.9 and
                disk_needed <= self.disk_pool * 0.9 and
                len(current_batch) < 8):  # Max 8 concurrent sessions
                
                current_batch.append(session_id)
                current_memory = memory_needed
                current_cpu = cpu_needed
                current_disk = disk_needed
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                
                current_batch = [session_id]
                current_memory = quota.max_memory_mb / 1024
                current_cpu = quota.max_cpu_percent
                current_disk = quota.max_disk_gb
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Recommended {len(batches)} batches for {len(resource_predictions)} sessions")
        return batches
    
    def _can_allocate(self, quota: ResourceQuota) -> bool:
        """Check if resources can be allocated."""
        memory_needed = self.allocated_memory + quota.max_memory_mb / 1024
        cpu_needed = self.allocated_cpu + quota.max_cpu_percent
        disk_needed = self.allocated_disk + quota.max_disk_gb
        
        return (memory_needed <= self.memory_pool and
                cpu_needed <= self.cpu_pool and
                disk_needed <= self.disk_pool)
    
    def _get_allocation_failure_reason(self, quota: ResourceQuota) -> str:
        """Get reason why allocation failed."""
        reasons = []
        
        memory_needed = self.allocated_memory + quota.max_memory_mb / 1024
        if memory_needed > self.memory_pool:
            reasons.append(f"Memory: need {memory_needed:.1f}GB, available {self.memory_pool:.1f}GB")
        
        cpu_needed = self.allocated_cpu + quota.max_cpu_percent
        if cpu_needed > self.cpu_pool:
            reasons.append(f"CPU: need {cpu_needed:.1f}%, available {self.cpu_pool:.1f}%")
        
        disk_needed = self.allocated_disk + quota.max_disk_gb
        if disk_needed > self.disk_pool:
            reasons.append(f"Disk: need {disk_needed:.1f}GB, available {self.disk_pool:.1f}GB")
        
        return "; ".join(reasons)
    
    def _calculate_network_complexity(self, network_config: Dict[str, Any]) -> float:
        """Calculate network complexity score for resource estimation."""
        complexity = 0.0
        
        # Base complexity from neuron count
        total_neurons = 0
        if 'layers' in network_config:
            for layer in network_config['layers']:
                total_neurons += layer.get('size', 0)
        
        # Logarithmic scaling for neuron count
        if total_neurons > 0:
            complexity += math.log10(total_neurons) / 6.0  # Normalize to ~1.0 for 1M neurons
        
        # Connection complexity
        if 'connections' in network_config:
            connection_count = len(network_config['connections'])
            complexity += math.log10(max(1, connection_count)) / 4.0
        
        # Layer depth complexity
        layer_count = len(network_config.get('layers', []))
        complexity += layer_count / 20.0  # Normalize to ~1.0 for 20 layers
        
        # Special features complexity
        if 'plasticity' in network_config:
            complexity += 0.5  # STDP adds complexity
        
        if 'learning_rules' in network_config:
            complexity += len(network_config['learning_rules']) * 0.2
        
        return min(3.0, complexity)  # Cap at 3.0
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                # Update resource predictions based on historical data
                self._update_resource_predictions()
                
                # Check for resource violations
                self._check_resource_violations()
                
                # Cleanup old usage history
                self._cleanup_old_usage_data()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in resource manager monitor loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_resource_predictions(self):
        """Update resource predictions based on historical performance."""
        # Analyze usage patterns for each target
        target_performance = defaultdict(list)
        
        for session_usages in self.usage_history.values():
            if session_usages:
                # Get the session's target from FPGA allocations
                # This is a simplified approach - in practice, we'd track this more explicitly
                avg_duration = sum(u.duration_seconds for u in session_usages) / len(session_usages)
                target_performance['default'].append(avg_duration)
        
        # Update predictions
        for target, durations in target_performance.items():
            if len(durations) >= 5:  # Need sufficient data
                avg_actual = sum(durations) / len(durations)
                predicted = 120.0  # Base prediction
                
                adjustment_factor = avg_actual / max(1, predicted)
                self.resource_predictions[target] = adjustment_factor
    
    def _check_resource_violations(self):
        """Check for resource quota violations."""
        current_time = time.time()
        
        for session_id, quota in list(self.active_sessions.items()):
            if session_id in self.usage_history:
                latest_usage = self.usage_history[session_id][-1] if self.usage_history[session_id] else None
                
                if latest_usage:
                    # Check for violations
                    violations = []
                    
                    if latest_usage.memory_mb > quota.max_memory_mb:
                        violations.append(f"Memory: {latest_usage.memory_mb:.1f}MB > {quota.max_memory_mb}MB")
                    
                    if latest_usage.cpu_percent > quota.max_cpu_percent:
                        violations.append(f"CPU: {latest_usage.cpu_percent:.1f}% > {quota.max_cpu_percent:.1f}%")
                    
                    if latest_usage.duration_seconds > quota.max_duration_seconds:
                        violations.append(f"Duration: {latest_usage.duration_seconds}s > {quota.max_duration_seconds}s")
                    
                    if violations:
                        logger.warning(f"Resource violations for session {session_id}: {'; '.join(violations)}")
    
    def _cleanup_old_usage_data(self):
        """Remove old usage data to prevent memory growth."""
        cutoff_time = time.time() - 7200  # Keep 2 hours of data
        
        for session_id in list(self.usage_history.keys()):
            self.usage_history[session_id] = [
                usage for usage in self.usage_history[session_id]
                if usage.timestamp > cutoff_time
            ]
            
            # Remove empty histories
            if not self.usage_history[session_id]:
                del self.usage_history[session_id]