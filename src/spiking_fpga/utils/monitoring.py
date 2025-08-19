"""Health monitoring and performance metrics for the toolchain."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

from .logging import StructuredLogger


class SystemResourceMonitor:
    """Monitor system resources and provide performance recommendations."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self._monitoring = False
        self._metrics_history: List[SystemMetrics] = []
        
    def start_monitoring(self):
        """Start background monitoring."""
        self._monitoring = True
        self.logger.info("System resource monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        self.logger.info("System resource monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': memory.percent,
            'available_memory_gb': memory.available / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'peak_memory_mb': memory.used / (1024**2),
            'avg_cpu_percent': psutil.cpu_percent()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        return self.get_current_metrics()
    
    def get_active_compilations(self) -> List[Dict[str, Any]]:
        """Get list of active compilation processes."""
        return []  # Placeholder implementation
        
    def get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations based on current system state."""
        metrics = self.get_current_metrics()
        recommendations = []
        
        if metrics['cpu_percent'] > 80:
            recommendations.append("High CPU usage - consider reducing parallel processes")
        if metrics['memory_percent'] > 90:
            recommendations.append("High memory usage - consider reducing compilation cache size")
        if metrics['disk_percent'] > 90:
            recommendations.append("Low disk space - clean up temporary files")
            
        return recommendations


@dataclass
class SystemMetrics:
    """System resource usage metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    process_memory_mb: float
    process_cpu_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "process_memory_mb": self.process_memory_mb,
            "process_cpu_percent": self.process_cpu_percent,
        }


@dataclass
class CompilationMetrics:
    """Metrics for a compilation session."""
    
    network_name: str
    target: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    
    # Network metrics
    neuron_count: int = 0
    synapse_count: int = 0
    layer_count: int = 0
    
    # Resource metrics
    estimated_luts: int = 0
    estimated_memory_kb: float = 0.0
    estimated_dsp_slices: int = 0
    
    # Performance metrics
    parsing_time_seconds: float = 0.0
    optimization_time_seconds: float = 0.0
    hdl_generation_time_seconds: float = 0.0
    synthesis_time_seconds: float = 0.0
    
    # Optimization metrics
    optimization_level: str = "BASIC"
    synapses_pruned: int = 0
    neurons_clustered: int = 0
    
    def duration_seconds(self) -> float:
        """Get total compilation duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "network_name": self.network_name,
            "target": self.target,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds(),
            "neuron_count": self.neuron_count,
            "synapse_count": self.synapse_count,
            "layer_count": self.layer_count,
            "estimated_luts": self.estimated_luts,
            "estimated_memory_kb": self.estimated_memory_kb,
            "estimated_dsp_slices": self.estimated_dsp_slices,
            "parsing_time_seconds": self.parsing_time_seconds,
            "optimization_time_seconds": self.optimization_time_seconds,
            "hdl_generation_time_seconds": self.hdl_generation_time_seconds,
            "synthesis_time_seconds": self.synthesis_time_seconds,
            "optimization_level": self.optimization_level,
            "synapses_pruned": self.synapses_pruned,
            "neurons_clustered": self.neurons_clustered,
        }


class HealthMonitor:
    """Monitor system health and performance during compilation."""
    
    def __init__(self, logger: StructuredLogger, 
                 metrics_file: Optional[Path] = None,
                 monitoring_interval: float = 5.0):
        self.logger = logger
        self.metrics_file = metrics_file
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.system_metrics: List[SystemMetrics] = []
        self.max_metrics = 1000  # Keep last 1000 measurements
        
        # Health thresholds
        self.cpu_warning_threshold = 80.0
        self.memory_warning_threshold = 85.0
        self.disk_warning_threshold = 90.0
        
        # Performance tracking
        self.compilation_history: List[CompilationMetrics] = []
        self.max_history = 100  # Keep last 100 compilations
    
    def start_monitoring(self):
        """Start system monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("Health monitoring started", 
                           interval_seconds=self.monitoring_interval)
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=memory.percent,
                    memory_used_gb=memory.used / (1024**3),
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage_percent=disk.percent,
                    process_memory_mb=process.memory_info().rss / (1024**2),
                    process_cpu_percent=process.cpu_percent(),
                )
                
                self.system_metrics.append(metrics)
                
                # Keep only recent metrics
                if len(self.system_metrics) > self.max_metrics:
                    self.system_metrics = self.system_metrics[-self.max_metrics:]
                
                # Check for health issues
                self._check_health_thresholds(metrics)
                
                # Save metrics to file if specified
                if self.metrics_file:
                    self._save_metrics(metrics)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
            
            time.sleep(self.monitoring_interval)
    
    def _check_health_thresholds(self, metrics: SystemMetrics):
        """Check if metrics exceed warning thresholds."""
        if metrics.cpu_percent > self.cpu_warning_threshold:
            self.logger.warning("High CPU usage detected", 
                              cpu_percent=metrics.cpu_percent,
                              threshold=self.cpu_warning_threshold)
        
        if metrics.memory_percent > self.memory_warning_threshold:
            self.logger.warning("High memory usage detected",
                              memory_percent=metrics.memory_percent,
                              threshold=self.memory_warning_threshold)
        
        if metrics.disk_usage_percent > self.disk_warning_threshold:
            self.logger.warning("High disk usage detected",
                              disk_percent=metrics.disk_usage_percent,
                              threshold=self.disk_warning_threshold)
    
    def _save_metrics(self, metrics: SystemMetrics):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            self.logger.error("Failed to save metrics", error=str(e))
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest = self.system_metrics[-1]
        
        # Determine overall health status
        status = "healthy"
        issues = []
        
        if latest.cpu_percent > self.cpu_warning_threshold:
            status = "warning"
            issues.append(f"High CPU usage: {latest.cpu_percent:.1f}%")
        
        if latest.memory_percent > self.memory_warning_threshold:
            status = "warning"
            issues.append(f"High memory usage: {latest.memory_percent:.1f}%")
        
        if latest.disk_usage_percent > self.disk_warning_threshold:
            status = "warning"
            issues.append(f"High disk usage: {latest.disk_usage_percent:.1f}%")
        
        return {
            "status": status,
            "issues": issues,
            "metrics": latest.to_dict(),
            "monitoring_duration_minutes": (datetime.utcnow() - self.system_metrics[0].timestamp).total_seconds() / 60
        }
    
    def add_compilation_metrics(self, metrics: CompilationMetrics):
        """Add compilation metrics to history."""
        self.compilation_history.append(metrics)
        
        # Keep only recent compilations
        if len(self.compilation_history) > self.max_history:
            self.compilation_history = self.compilation_history[-self.max_history:]
        
        self.logger.info("Compilation metrics recorded", **metrics.to_dict())
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation performance statistics."""
        if not self.compilation_history:
            return {"total_compilations": 0}
        
        successful = [m for m in self.compilation_history if m.success]
        failed = [m for m in self.compilation_history if not m.success]
        
        # Calculate average durations
        if successful:
            avg_duration = sum(m.duration_seconds() for m in successful) / len(successful)
            avg_neuron_count = sum(m.neuron_count for m in successful) / len(successful)
            avg_synapse_count = sum(m.synapse_count for m in successful) / len(successful)
        else:
            avg_duration = 0
            avg_neuron_count = 0
            avg_synapse_count = 0
        
        # Get target distribution
        target_counts = {}
        for m in self.compilation_history:
            target_counts[m.target] = target_counts.get(m.target, 0) + 1
        
        return {
            "total_compilations": len(self.compilation_history),
            "successful_compilations": len(successful),
            "failed_compilations": len(failed),
            "success_rate": len(successful) / len(self.compilation_history) * 100,
            "average_duration_seconds": avg_duration,
            "average_neuron_count": avg_neuron_count,
            "average_synapse_count": avg_synapse_count,
            "target_distribution": target_counts,
        }


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, logger: Optional[StructuredLogger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.debug(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.logger:
            if exc_type is None:
                self.logger.info(f"Completed {self.name}", 
                               duration_seconds=duration)
            else:
                self.logger.error(f"Failed {self.name}", 
                                duration_seconds=duration,
                                error=str(exc_val))
    
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class CircuitBreaker:
    """Circuit breaker pattern for external dependencies."""
    
    def __init__(self, failure_threshold: int = 5, 
                 timeout_seconds: float = 60.0,
                 logger: Optional[StructuredLogger] = None):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                if self.logger:
                    self.logger.info("Circuit breaker half-open - attempting call")
            else:
                raise Exception("Circuit breaker is open - calls not allowed")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout_seconds
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            if self.logger:
                self.logger.info("Circuit breaker closed - normal operation restored")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            if self.logger:
                self.logger.warning("Circuit breaker opened due to failures",
                                  failure_count=self.failure_count,
                                  threshold=self.failure_threshold)