"""Generation 3 Optimized Compiler with advanced performance features."""

import asyncio
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import psutil

from .generation2_robust_compiler import Generation2RobustCompiler, RobustCompilationConfig
from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel, ResourceEstimate
from .network_compiler import CompilationResult
from .performance.advanced_performance_optimization import (
    PerformanceProfiler, CacheManager, ResourceOptimizer, 
    AdaptiveOptimizationController, ParallelCompilationEngine
)
from .scalability.distributed_compiler import DistributedCompiler
from .scalability.auto_scaler import AutoScaler, ScalingPolicy
from .scalability.load_balancer import LoadBalancer
from .utils import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizedCompilationConfig(RobustCompilationConfig):
    """Advanced compilation configuration with performance optimization."""
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_worker_processes: int = 0  # 0 = auto-detect
    enable_caching: bool = True
    cache_size_mb: int = 512
    enable_adaptive_optimization: bool = True
    
    # Distributed compilation settings
    enable_distributed_compilation: bool = False
    distributed_workers: List[str] = field(default_factory=list)
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, weighted
    
    # Auto-scaling settings
    enable_auto_scaling: bool = True
    cpu_threshold: float = 80.0  # CPU usage percentage
    memory_threshold: float = 85.0  # Memory usage percentage
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    
    # Advanced caching
    enable_result_caching: bool = True
    cache_hdl_templates: bool = True
    cache_optimization_results: bool = True
    cache_resource_estimates: bool = True
    
    # Performance monitoring
    enable_performance_profiling: bool = True
    profile_memory_usage: bool = True
    profile_cpu_usage: bool = True
    benchmark_mode: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_worker_processes <= 0:
            self.max_worker_processes = max(1, mp.cpu_count() - 1)


class Generation3OptimizedCompiler(Generation2RobustCompiler):
    """High-performance compiler with advanced optimization and scaling."""
    
    def __init__(self, target: FPGATarget,
                 config: Optional[OptimizedCompilationConfig] = None,
                 log_level: str = "INFO",
                 log_file: Optional[Path] = None):
        
        # Initialize configuration
        self.opt_config = config or OptimizedCompilationConfig()
        
        # Initialize parent with robust features
        super().__init__(target, self.opt_config, log_level, log_file)
        
        # Performance optimization components  
        if self.opt_config.enable_performance_profiling:
            self.performance_profiler = PerformanceProfiler()
        else:
            self.performance_profiler = None
        self.cache_manager = CacheManager(
            max_size_mb=self.opt_config.cache_size_mb,
            enabled=self.opt_config.enable_caching
        )
        self.resource_optimizer = ResourceOptimizer()
        
        # Adaptive optimization
        if self.opt_config.enable_adaptive_optimization:
            self.adaptive_controller = AdaptiveOptimizationController(
                profiler=self.performance_profiler,
                cache_manager=self.cache_manager
            )
        else:
            self.adaptive_controller = None
            
        # Parallel processing
        if self.opt_config.enable_parallel_processing:
            self.parallel_engine = ParallelCompilationEngine(
                max_workers=self.opt_config.max_worker_processes
            )
        else:
            self.parallel_engine = None
            
        # Distributed compilation
        if self.opt_config.enable_distributed_compilation:
            self.distributed_compiler = DistributedCompiler(
                workers=self.opt_config.distributed_workers,
                load_balancer=LoadBalancer(strategy=self.opt_config.load_balancing_strategy)
            )
        else:
            self.distributed_compiler = None
            
        # Auto-scaling
        if self.opt_config.enable_auto_scaling:
            from .scalability.auto_scaler import ScalingConfig
            scaling_config = ScalingConfig(
                scale_up_threshold=self.opt_config.cpu_threshold,
                memory_threshold=self.opt_config.memory_threshold,
                scale_up_cooldown=self.opt_config.scale_up_cooldown_seconds,
                scale_down_cooldown=self.opt_config.scale_down_cooldown_seconds
            )
            self.auto_scaler = AutoScaler(scaling_config)
        else:
            self.auto_scaler = None
            
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gen3_compiler")
        
        logger.info("Generation3OptimizedCompiler initialized", extra={
            "target": target.value,
            "parallel_processing": self.opt_config.enable_parallel_processing,
            "distributed_compilation": self.opt_config.enable_distributed_compilation,
            "auto_scaling": self.opt_config.enable_auto_scaling,
            "max_workers": self.opt_config.max_worker_processes
        })
        
    def compile_optimized(self, network: Union[Network, str, Path, Dict[str, Any]],
                         output_dir: Union[str, Path],
                         config: Optional[OptimizedCompilationConfig] = None) -> CompilationResult:
        """Compile with advanced performance optimization."""
        
        if config is None:
            config = self.opt_config
            
        start_time = time.time()
        
        try:
            # Check cache first
            if config.enable_result_caching:
                cached_result = self._check_compilation_cache(network, config)
                if cached_result:
                    logger.info("Returning cached compilation result")
                    return cached_result
                    
            # Auto-scaling check
            if self.auto_scaler:
                self.auto_scaler.check_and_scale()
                
            # Choose compilation strategy based on network size and resources
            compilation_strategy = self._select_compilation_strategy(network, config)
            
            if compilation_strategy == "distributed":
                result = self._compile_distributed(network, output_dir, config)
            elif compilation_strategy == "parallel":
                result = self._compile_parallel(network, output_dir, config)
            else:
                result = self._compile_sequential(network, output_dir, config)
                
            # Cache successful results
            if result.success and config.enable_result_caching:
                self._cache_compilation_result(network, config, result)
                
            # Update adaptive optimization
            if self.adaptive_controller:
                compilation_time = time.time() - start_time
                self.adaptive_controller.update_performance_metrics(
                    compilation_time=compilation_time,
                    success=result.success,
                    network_size=self._estimate_network_complexity(network)
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Optimized compilation failed: {e}")
            # Fallback to robust compilation
            logger.info("Falling back to robust compilation")
            return super().compile_robust(network, output_dir, config)
                
    def _select_compilation_strategy(self, network: Any, config: OptimizedCompilationConfig) -> str:
        """Select optimal compilation strategy based on network and system state."""
        
        network_complexity = self._estimate_network_complexity(network)
        system_resources = self._get_system_resources()
        
        logger.debug(f"Network complexity: {network_complexity}, System CPU: {system_resources['cpu_percent']}%")
        
        # Use distributed compilation for very large networks if available
        if (config.enable_distributed_compilation and 
            self.distributed_compiler and 
            network_complexity > 50000):  # Large network threshold
            return "distributed"
            
        # Use parallel compilation for medium networks with sufficient resources
        if (config.enable_parallel_processing and 
            self.parallel_engine and 
            network_complexity > 1000 and 
            system_resources['cpu_percent'] < 70):
            return "parallel"
            
        # Sequential compilation for small networks or resource-constrained systems
        return "sequential"
        
    def _estimate_network_complexity(self, network: Any) -> int:
        """Estimate network complexity for strategy selection."""
        if isinstance(network, Network):
            return len(network.neurons) + len(network.synapses)
        elif isinstance(network, dict):
            # Rough estimation from dict structure
            total_neurons = sum(layer.get('size', 0) for layer in network.get('layers', []))
            estimated_connections = total_neurons * 10  # Rough estimate
            return total_neurons + estimated_connections
        else:
            return 1000  # Default assumption
            
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource utilization."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get system resources: {e}")
            return {'cpu_percent': 50.0, 'memory_percent': 50.0, 'available_memory_gb': 4.0}
            
    def _compile_distributed(self, network: Any, output_dir: Any, 
                           config: OptimizedCompilationConfig) -> CompilationResult:
        """Compile using distributed processing."""
        logger.info("Starting distributed compilation")
        
        if not self.distributed_compiler:
            raise RuntimeError("Distributed compiler not initialized")
            
        return self.distributed_compiler.compile_distributed(
            network=network,
            output_dir=output_dir,
            config=config,
            target=self.target
        )
        
    def _compile_parallel(self, network: Any, output_dir: Any,
                         config: OptimizedCompilationConfig) -> CompilationResult:
        """Compile using parallel processing."""
        logger.info("Starting parallel compilation")
        
        if not self.parallel_engine:
            raise RuntimeError("Parallel engine not initialized")
            
        return self.parallel_engine.compile_parallel(
            network=network,
            output_dir=output_dir,
            config=config,
            base_compiler=self
        )
        
    def _compile_sequential(self, network: Any, output_dir: Any,
                          config: OptimizedCompilationConfig) -> CompilationResult:
        """Compile using sequential processing with optimizations."""
        logger.info("Starting sequential compilation with optimizations")
        
        # Apply resource optimization
        if isinstance(network, Network):
            optimized_network = self.resource_optimizer.optimize_network(network)
        else:
            optimized_network = network
            
        return super().compile_robust(optimized_network, output_dir, config)
        
    def _check_compilation_cache(self, network: Any, config: OptimizedCompilationConfig) -> Optional[CompilationResult]:
        """Check if compilation result is cached."""
        if not config.enable_result_caching:
            return None
            
        cache_key = self._generate_cache_key(network, config)
        return self.cache_manager.get(cache_key)
        
    def _cache_compilation_result(self, network: Any, config: OptimizedCompilationConfig, 
                                 result: CompilationResult):
        """Cache compilation result."""
        cache_key = self._generate_cache_key(network, config)
        self.cache_manager.put(cache_key, result)
        
    def _generate_cache_key(self, network: Any, config: OptimizedCompilationConfig) -> str:
        """Generate cache key for network and config."""
        import hashlib
        
        # Create hashable representation (using secure hashing)
        if isinstance(network, Network):
            network_hash = hashlib.sha256(str(network.name + str(len(network.neurons))).encode()).hexdigest()[:16]
        elif isinstance(network, dict):
            network_hash = hashlib.sha256(str(sorted(network.items())).encode()).hexdigest()[:16]
        else:
            network_hash = hashlib.sha256(str(network).encode()).hexdigest()[:16]
            
        config_hash = hashlib.sha256(str(config.optimization_level).encode()).hexdigest()[:16]
        target_hash = hashlib.sha256(self.target.value.encode()).hexdigest()[:16]
        
        return f"compilation_{network_hash}_{config_hash}_{target_hash}"
        
    async def compile_optimized_async(self, network: Union[Network, str, Path, Dict[str, Any]],
                                     output_dir: Union[str, Path],
                                     config: Optional[OptimizedCompilationConfig] = None) -> CompilationResult:
        """Asynchronous compilation with optimization."""
        loop = asyncio.get_event_loop()
        
        # Run compilation in thread pool to avoid blocking
        return await loop.run_in_executor(
            self.thread_pool,
            self.compile_optimized,
            network,
            output_dir,
            config
        )
        
    def batch_compile_optimized(self, networks: List[Tuple[Any, str]], 
                              config: Optional[OptimizedCompilationConfig] = None) -> List[CompilationResult]:
        """Batch compilation with optimization."""
        logger.info(f"Starting batch compilation of {len(networks)} networks")
        
        results = []
        
        if self.parallel_engine and config and config.enable_parallel_processing:
            # Use parallel processing for batch compilation
            with ProcessPoolExecutor(max_workers=self.opt_config.max_worker_processes) as executor:
                futures = []
                for network, output_dir in networks:
                    future = executor.submit(self.compile_optimized, network, output_dir, config)
                    futures.append(future)
                    
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per compilation
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch compilation failed for one network: {e}")
                        # Create error result
                        error_result = CompilationResult(
                            success=False,
                            network=Network(name="failed"),
                            optimized_network=Network(name="failed"),
                            hdl_files={},
                            resource_estimate=ResourceEstimate(0, 0, 0, 0, 0, 0),
                            optimization_stats={},
                            errors=[str(e)]
                        )
                        results.append(error_result)
        else:
            # Sequential batch compilation
            for network, output_dir in networks:
                try:
                    result = self.compile_optimized(network, output_dir, config)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch compilation failed for one network: {e}")
                    error_result = CompilationResult(
                        success=False,
                        network=Network(name="failed"),
                        optimized_network=Network(name="failed"),
                        hdl_files={},
                        resource_estimate=ResourceEstimate(0, 0, 0, 0, 0, 0),
                        optimization_stats={},
                        errors=[str(e)]
                    )
                    results.append(error_result)
                    
        successful_compilations = sum(1 for r in results if r.success)
        logger.info(f"Batch compilation completed: {successful_compilations}/{len(networks)} successful")
        
        return results
        
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = super().get_robustness_statistics()
        stats["compiler_type"] = "Generation3OptimizedCompiler"
        
        # Performance stats
        if self.performance_profiler:
            stats["performance"] = self.performance_profiler.get_statistics()
            
        # Cache stats
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_statistics()
            
        # Auto-scaler stats
        if self.auto_scaler:
            stats["auto_scaling"] = self.auto_scaler.get_statistics()
            
        # System resource stats
        stats["system_resources"] = self._get_system_resources()
        
        return stats
        
    def cleanup(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            
        if self.parallel_engine:
            self.parallel_engine.cleanup()
            
        if self.distributed_compiler:
            self.distributed_compiler.cleanup()
            
        if self.auto_scaler:
            self.auto_scaler.stop()
            
        logger.info("Generation3OptimizedCompiler cleanup completed")


def compile_network_optimized(network: Union[Network, str, Path, Dict[str, Any]],
                             target: FPGATarget,
                             output_dir: Union[str, Path] = "./output_gen3",
                             optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
                             enable_parallel: bool = True,
                             enable_caching: bool = True,
                             enable_auto_scaling: bool = True,
                             max_workers: int = 0) -> CompilationResult:
    """Compile with Generation 3 optimization features.
    
    Args:
        network: Network object, path to network definition file, or network dictionary
        target: Target FPGA platform
        output_dir: Directory for generated files
        optimization_level: Level of optimization to apply
        enable_parallel: Enable parallel processing
        enable_caching: Enable result caching
        enable_auto_scaling: Enable auto-scaling
        max_workers: Maximum worker processes (0 = auto-detect)
    
    Returns:
        CompilationResult with generated files and statistics
    """
    config = OptimizedCompilationConfig(
        optimization_level=optimization_level,
        enable_parallel_processing=enable_parallel,
        enable_caching=enable_caching,
        enable_auto_scaling=enable_auto_scaling,
        max_worker_processes=max_workers
    )
    
    compiler = Generation3OptimizedCompiler(target, config)
    try:
        return compiler.compile_optimized(network, output_dir, config)
    finally:
        compiler.cleanup()