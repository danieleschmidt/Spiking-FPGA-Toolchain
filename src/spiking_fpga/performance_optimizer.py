"""Advanced performance optimization module for production-scale FPGA compilation."""

import asyncio
import concurrent.futures
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import psutil
import threading
from datetime import datetime, timedelta

from .scalable_compiler import ScalableNetworkCompiler, ScalableCompilationConfig
from .core import FPGATarget
from .utils import StructuredLogger, configure_logging


@dataclass
class PerformanceProfile:
    """Performance characteristics for different workload types."""
    
    name: str
    target_latency_ms: float
    target_throughput_networks_per_min: float
    max_memory_usage_mb: float
    max_cpu_percent: float
    cache_strategy: str = "aggressive"
    concurrency_level: int = 4


class AdaptivePerformanceController:
    """Dynamically optimizes compilation performance based on system resources and workload."""
    
    def __init__(self):
        self.logger = configure_logging("INFO")
        from .utils.monitoring import SystemResourceMonitor
        self.system_monitor = SystemResourceMonitor()
        self.performance_history: List[Dict[str, Any]] = []
        self.current_profile = self._get_default_profile()
        
        # Performance profiles for different scenarios
        self.profiles = {
            "development": PerformanceProfile(
                name="development",
                target_latency_ms=2000.0,
                target_throughput_networks_per_min=10.0,
                max_memory_usage_mb=1024.0,
                max_cpu_percent=50.0,
                concurrency_level=2
            ),
            "production": PerformanceProfile(
                name="production", 
                target_latency_ms=500.0,
                target_throughput_networks_per_min=100.0,
                max_memory_usage_mb=8192.0,
                max_cpu_percent=90.0,
                concurrency_level=8
            ),
            "batch": PerformanceProfile(
                name="batch",
                target_latency_ms=10000.0,
                target_throughput_networks_per_min=500.0,
                max_memory_usage_mb=16384.0,
                max_cpu_percent=95.0,
                concurrency_level=16
            ),
        }
    
    def _get_default_profile(self) -> PerformanceProfile:
        """Select default profile based on system capabilities."""
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 16 and cpu_count >= 8:
            return self.profiles["production"]
        elif memory_gb >= 8 and cpu_count >= 4:
            return self.profiles["development"]
        else:
            # Conservative profile for limited resources
            return PerformanceProfile(
                name="conservative",
                target_latency_ms=5000.0,
                target_throughput_networks_per_min=5.0,
                max_memory_usage_mb=512.0,
                max_cpu_percent=30.0,
                concurrency_level=1
            )
    
    def optimize_configuration(self, workload_type: str = "development") -> ScalableCompilationConfig:
        """Generate optimized configuration based on current conditions."""
        profile = self.profiles.get(workload_type, self.current_profile)
        system_resources = self.system_monitor.get_current_metrics()
        
        # Adjust concurrency based on available resources
        available_memory_gb = system_resources["available_memory_gb"]
        cpu_usage_percent = system_resources.get("cpu_percent", 0)
        
        # Dynamic concurrency adjustment
        optimal_workers = min(
            profile.concurrency_level,
            max(1, int(available_memory_gb / 2)),  # 2GB per worker
            max(1, psutil.cpu_count() - 1)  # Leave one core free
        )
        
        # Reduce workers if system is under load
        if cpu_usage_percent > 80:
            optimal_workers = max(1, optimal_workers // 2)
        
        # Cache configuration
        cache_size_mb = min(
            profile.max_memory_usage_mb * 0.3,  # 30% of budget for cache
            available_memory_gb * 1024 * 0.1   # 10% of available memory
        )
        
        config = ScalableCompilationConfig(
            enable_caching=True,
            cache_dir=Path.home() / ".spiking_fpga_cache",
            enable_concurrency=optimal_workers > 1,
            max_concurrent_workers=optimal_workers,
            use_load_balancer=optimal_workers > 2,
            cache_ttl_hours=24.0 if workload_type == "development" else 72.0
        )
        
        self.logger.info("Performance configuration optimized", 
                        profile=profile.name,
                        optimal_workers=optimal_workers,
                        cache_size_mb=cache_size_mb,
                        available_memory_gb=available_memory_gb,
                        cpu_usage=cpu_usage_percent)
        
        return config
    
    def benchmark_compiler(self, compiler: ScalableNetworkCompiler, 
                          test_networks: List[Any], 
                          target: FPGATarget) -> Dict[str, Any]:
        """Comprehensive performance benchmarking."""
        self.logger.info("Starting performance benchmark", 
                        network_count=len(test_networks))
        
        results = {
            "benchmark_start": datetime.utcnow().isoformat(),
            "system_info": self.system_monitor.get_current_metrics(),
            "test_results": [],
            "aggregate_metrics": {}
        }
        
        durations = []
        success_count = 0
        
        for i, network in enumerate(test_networks):
            self.logger.info(f"Benchmarking network {i+1}/{len(test_networks)}")
            
            # Measure system resources before
            pre_resources = self.system_monitor.get_current_metrics()
            
            start_time = time.perf_counter()
            
            try:
                result = compiler.compile(
                    network,
                    target,
                    Path(f"./benchmark_output_{i}"),
                )
                
                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                durations.append(duration)
                
                if result.success:
                    success_count += 1
                
                # Measure system resources after
                post_resources = self.system_monitor.get_current_metrics()
                
                test_result = {
                    "test_id": i,
                    "duration_ms": duration,
                    "success": result.success,
                    "estimated_luts": result.resource_estimate.luts,
                    "estimated_memory_kb": result.resource_estimate.bram_kb,
                    "pre_cpu_percent": pre_resources["cpu_usage_percent"],
                    "post_cpu_percent": post_resources["cpu_usage_percent"],
                    "memory_delta_mb": post_resources["used_memory_gb"] - pre_resources["used_memory_gb"]
                }
                
                results["test_results"].append(test_result)
                
            except Exception as e:
                self.logger.error(f"Benchmark test {i} failed", error=str(e))
                results["test_results"].append({
                    "test_id": i,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate aggregate metrics
        if durations:
            results["aggregate_metrics"] = {
                "total_tests": len(test_networks),
                "successful_tests": success_count,
                "success_rate_percent": (success_count / len(test_networks)) * 100,
                "average_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "throughput_networks_per_minute": 60000 / (sum(durations) / len(durations)) if durations else 0,
                "p95_latency_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations) if durations else 0
            }
        
        results["benchmark_end"] = datetime.utcnow().isoformat()
        
        self.logger.info("Performance benchmark completed", 
                        **results["aggregate_metrics"])
        
        return results
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze performance metrics and provide optimization suggestions."""
        suggestions = []
        
        cpu_percent = metrics.get('cpu_percent', 0)
        memory_percent = metrics.get('memory_percent', 0)
        available_memory = metrics.get('available_memory_gb', 0)
        
        if cpu_percent > 80:
            suggestions.append("High CPU usage detected - consider reducing parallel compilation tasks")
        
        if memory_percent > 85:
            suggestions.append("High memory usage - consider reducing compilation cache size")
        
        if available_memory < 1:
            suggestions.append("Low available memory - close unnecessary applications")
            
        if cpu_percent < 20 and memory_percent < 50:
            suggestions.append("System resources underutilized - consider increasing parallel compilation")
        
        return suggestions
    
    def learn_from_benchmark(self, config: Dict[str, Any], metrics: Dict[str, Any], result: Any):
        """Learn from benchmark results to improve future optimizations."""
        self.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'config': config,
            'metrics': metrics,
            'success': getattr(result, 'success', False)
        })
        
        # Keep only last 100 entries to prevent memory growth
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def monitor_real_time_performance(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Monitor system performance in real-time."""
        self.logger.info("Starting real-time performance monitoring", 
                        duration_minutes=duration_minutes)
        
        monitoring_data = {
            "start_time": datetime.utcnow().isoformat(),
            "duration_minutes": duration_minutes,
            "samples": [],
            "alerts": []
        }
        
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        while datetime.utcnow() < end_time:
            sample = self.system_monitor.get_detailed_metrics()
            sample["timestamp"] = datetime.utcnow().isoformat()
            monitoring_data["samples"].append(sample)
            
            # Check for performance alerts
            alerts = self._check_performance_alerts(sample)
            monitoring_data["alerts"].extend(alerts)
            
            time.sleep(30)  # Sample every 30 seconds
        
        monitoring_data["end_time"] = datetime.utcnow().isoformat()
        
        # Generate summary statistics
        if monitoring_data["samples"]:
            cpu_values = [s["cpu_usage_percent"] for s in monitoring_data["samples"]]
            memory_values = [s["used_memory_gb"] for s in monitoring_data["samples"]]
            
            monitoring_data["summary"] = {
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                "max_cpu_percent": max(cpu_values),
                "avg_memory_gb": sum(memory_values) / len(memory_values),
                "max_memory_gb": max(memory_values),
                "total_alerts": len(monitoring_data["alerts"]),
                "samples_collected": len(monitoring_data["samples"])
            }
        
        self.logger.info("Real-time monitoring completed", 
                        **monitoring_data.get("summary", {}))
        
        return monitoring_data
    
    def _check_performance_alerts(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # High CPU usage alert
        if sample["cpu_usage_percent"] > 90:
            alerts.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"CPU usage is {sample['cpu_usage_percent']:.1f}%",
                "timestamp": sample["timestamp"]
            })
        
        # High memory usage alert
        if sample["memory_usage_percent"] > 85:
            alerts.append({
                "type": "high_memory",
                "severity": "warning", 
                "message": f"Memory usage is {sample['memory_usage_percent']:.1f}%",
                "timestamp": sample["timestamp"]
            })
        
        # Low available memory alert
        if sample["available_memory_gb"] < 1.0:
            alerts.append({
                "type": "low_memory",
                "severity": "critical",
                "message": f"Available memory is {sample['available_memory_gb']:.2f} GB",
                "timestamp": sample["timestamp"]
            })
        
        return alerts


class SystemResourceMonitor:
    """Monitor system resources for performance optimization."""
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        return {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": psutil.platform.system(),
            "python_version": psutil.sys.version
        }
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        memory = psutil.virtual_memory()
        
        return {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": memory.percent,
            "used_memory_gb": memory.used / (1024**3),
            "available_memory_gb": memory.available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics."""
        cpu_times = psutil.cpu_times_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage_percent": psutil.cpu_percent(),
            "cpu_user_percent": cpu_times.user,
            "cpu_system_percent": cpu_times.system,
            "cpu_idle_percent": cpu_times.idle,
            "memory_usage_percent": memory.percent,
            "used_memory_gb": memory.used / (1024**3),
            "available_memory_gb": memory.available / (1024**3),
            "cached_memory_gb": memory.cached / (1024**3),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
            "process_count": len(psutil.pids())
        }


def create_optimized_compiler(workload_type: str = "development") -> Tuple[ScalableNetworkCompiler, Dict[str, Any]]:
    """Factory function to create optimally configured compiler."""
    controller = AdaptivePerformanceController()
    config = controller.optimize_configuration(workload_type)
    compiler = ScalableNetworkCompiler(config)
    
    optimization_info = {
        "workload_type": workload_type,
        "configuration": {
            "enable_caching": config.enable_caching,
            "enable_concurrency": config.enable_concurrency,
            "max_concurrent_workers": config.max_concurrent_workers,
            "use_load_balancer": config.use_load_balancer,
            "cache_ttl_hours": config.cache_ttl_hours
        },
        "system_resources": controller.system_monitor.get_current_resources(),
        "recommended_profile": controller.current_profile.name
    }
    
    return compiler, optimization_info