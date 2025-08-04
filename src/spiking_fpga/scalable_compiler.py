"""Scalable, high-performance network compiler with caching and concurrency."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime

from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel, ResourceEstimate
from .network_compiler import NetworkCompiler, CompilationResult, CompilationConfig
from .utils import (
    StructuredLogger, configure_logging,
    CompilationCache, ConcurrentCompiler, AdaptiveLoadBalancer,
    PerformanceTimer, ResourcePool
)


@dataclass
class ScalableCompilationConfig(CompilationConfig):
    """Extended configuration for scalable compilation."""
    
    enable_caching: bool = True
    cache_dir: Optional[Path] = None
    enable_concurrency: bool = True
    max_concurrent_workers: Optional[int] = None
    use_load_balancer: bool = True
    cache_ttl_hours: float = 24.0


class ScalableNetworkCompiler:
    """High-performance, scalable network compiler with intelligent caching and concurrency."""
    
    def __init__(self, config: ScalableCompilationConfig = None):
        self.config = config or ScalableCompilationConfig()
        self.logger = configure_logging("INFO")
        
        # Initialize caching
        self.cache = None
        if self.config.enable_caching:
            cache_dir = self.config.cache_dir or Path.home() / ".spiking_fpga_cache"
            self.cache = CompilationCache(cache_dir, enable_memory_cache=True)
            self.logger.info("Caching enabled", cache_dir=str(cache_dir))
        
        # Initialize concurrency
        self.concurrent_compiler = None
        self.load_balancer = None
        
        if self.config.enable_concurrency:
            if self.config.use_load_balancer:
                initial_workers = min(4, self.config.max_concurrent_workers or 4)
                self.load_balancer = AdaptiveLoadBalancer(initial_workers=initial_workers)
                self.logger.info("Load balancer enabled", initial_workers=initial_workers)
            else:
                self.concurrent_compiler = ConcurrentCompiler(
                    max_workers=self.config.max_concurrent_workers
                )
                self.logger.info("Concurrent compiler enabled", 
                               max_workers=self.concurrent_compiler.max_workers)
        
        # Resource pool for expensive objects
        self.compiler_pool = ResourcePool(
            factory=lambda: NetworkCompiler(
                FPGATarget.ARTIX7_35T,  # Default target, will be overridden
                enable_monitoring=False,
                log_level="WARNING"  # Reduce log noise for pooled compilers
            ),
            max_size=8,
            min_size=2
        )
        
        self.logger.info("ScalableNetworkCompiler initialized",
                        caching_enabled=self.config.enable_caching,
                        concurrency_enabled=self.config.enable_concurrency,
                        load_balancer_enabled=self.config.use_load_balancer)
    
    def _generate_network_hash(self, network: Union[Network, Path]) -> str:
        """Generate a hash for network caching."""
        if isinstance(network, Path):
            # Hash file content and modification time
            content = network.read_text()
            mtime = network.stat().st_mtime
            data = f"{content}:{mtime}"
        else:
            # Hash network structure
            data = json.dumps({
                "name": network.name,
                "neurons": len(network.neurons),
                "synapses": len(network.synapses),
                "layers": [(layer.layer_type.value, layer.size, layer.neuron_type) 
                          for layer in network.layers],
            }, sort_keys=True)
        
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_config_hash(self, config: ScalableCompilationConfig, target: FPGATarget) -> str:
        """Generate a hash for compilation configuration."""
        config_data = {
            "target": target.value,
            "optimization_level": config.optimization_level.value,
            "clock_frequency": config.clock_frequency,
            "debug_enabled": config.debug_enabled,
            "run_synthesis": config.run_synthesis,
        }
        data = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def compile(self, network: Union[Network, str, Path], 
               target: FPGATarget,
               output_dir: Union[str, Path],
               config: ScalableCompilationConfig = None) -> CompilationResult:
        """Compile with intelligent caching and optimization."""
        if config is None:
            config = self.config
        
        network_path = Path(network) if isinstance(network, (str, Path)) else None
        output_dir = Path(output_dir)
        
        # Generate cache keys
        network_hash = self._generate_network_hash(network)
        config_hash = self._generate_config_hash(config, target)
        
        with PerformanceTimer("total_compilation", self.logger) as total_timer:
            # Try to get cached result
            if self.cache:
                with PerformanceTimer("cache_lookup", self.logger) as cache_timer:
                    cached_result = self.cache.get_optimization_result(network_hash, config_hash)
                    if cached_result:
                        self.logger.info("Cache hit - using cached compilation result",
                                       network_hash=network_hash,
                                       config_hash=config_hash,
                                       cache_lookup_time=cache_timer.elapsed_seconds())
                        
                        # Recreate result object with updated paths
                        return self._reconstruct_cached_result(cached_result, output_dir)
            
            # Cache miss - perform compilation
            self.logger.info("Cache miss - performing fresh compilation",
                           network_hash=network_hash,
                           config_hash=config_hash)
            
            # Get compiler from pool
            with self.compiler_pool.get_resource() as compiler:
                # Update compiler target
                compiler.target = target
                
                # Perform compilation
                with PerformanceTimer("compilation", self.logger) as comp_timer:
                    result = compiler.compile(network, output_dir, config)
                
                # Cache successful results
                if result.success and self.cache:
                    with PerformanceTimer("cache_store", self.logger) as store_timer:
                        cache_data = self._prepare_result_for_caching(result)
                        self.cache.put_optimization_result(
                            network_hash, 
                            config_hash, 
                            cache_data
                        )
                        self.logger.info("Cached compilation result",
                                       cache_store_time=store_timer.elapsed_seconds())
                
                return result
    
    def compile_concurrent(self, tasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Submit multiple compilation tasks for concurrent execution."""
        if not (self.concurrent_compiler or self.load_balancer):
            raise RuntimeError("Concurrency not enabled")
        
        task_ids = {}
        
        for i, task in enumerate(tasks):
            task_name = task.get("name", f"task_{i}")
            
            if self.load_balancer:
                task_id = self.load_balancer.submit_task(
                    network_path=Path(task["network"]),
                    target=task["target"],
                    output_dir=Path(task["output_dir"]),
                    optimization_level=task.get("optimization_level", 1),
                    **task.get("extra_params", {})
                )
            else:
                task_id = self.concurrent_compiler.submit_compilation(
                    network_path=Path(task["network"]),
                    target=task["target"],
                    output_dir=Path(task["output_dir"]),
                    optimization_level=task.get("optimization_level", 1),
                    **task.get("extra_params", {})
                )
            
            task_ids[task_name] = task_id
        
        self.logger.info("Submitted concurrent compilation tasks", 
                        task_count=len(tasks))
        
        return task_ids
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of concurrent task."""
        if self.load_balancer:
            return self.load_balancer.get_task_status(task_id)
        elif self.concurrent_compiler:
            return self.concurrent_compiler.get_task_status(task_id)
        else:
            return {"status": "not_found", "error": "Concurrency not enabled"}
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for a concurrent task to complete."""
        if self.concurrent_compiler:
            return self.concurrent_compiler.wait_for_task(task_id, timeout)
        else:
            return None
    
    def _prepare_result_for_caching(self, result: CompilationResult) -> Dict[str, Any]:
        """Prepare compilation result for caching."""
        return {
            "success": result.success,
            "resource_estimate": {
                "neurons": result.resource_estimate.neurons,
                "synapses": result.resource_estimate.synapses,
                "luts": result.resource_estimate.luts,
                "registers": result.resource_estimate.registers,
                "bram_kb": result.resource_estimate.bram_kb,
                "dsp_slices": result.resource_estimate.dsp_slices,
            },
            "optimization_stats": result.optimization_stats,
            "warnings": result.warnings,
            "errors": result.errors,
            "cached_at": datetime.utcnow().isoformat(),
        }
    
    def _reconstruct_cached_result(self, cached_data: Dict[str, Any], 
                                  output_dir: Path) -> CompilationResult:
        """Reconstruct CompilationResult from cached data."""
        # Create resource estimate
        resource_data = cached_data["resource_estimate"]
        resource_estimate = ResourceEstimate(
            luts=resource_data["luts"],
            registers=resource_data["registers"],
            bram_kb=resource_data["bram_kb"],
            dsp_slices=resource_data["dsp_slices"],
            neurons=resource_data["neurons"],
            synapses=resource_data["synapses"],
        )
        
        # Create dummy network objects (cached results don't need full network data)
        dummy_network = Network(name="cached_network")
        
        return CompilationResult(
            success=cached_data["success"],
            network=dummy_network,
            optimized_network=dummy_network,
            hdl_files={},  # HDL files would need to be regenerated
            resource_estimate=resource_estimate,
            optimization_stats=cached_data["optimization_stats"],
            warnings=cached_data.get("warnings", []),
            errors=cached_data.get("errors", []),
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "compiler_pool": self.compiler_pool.get_stats(),
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        if self.load_balancer:
            stats["load_balancer"] = self.load_balancer.get_stats()
        elif self.concurrent_compiler:
            stats["concurrent_compiler"] = self.concurrent_compiler.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance and cleanup."""
        if not self.cache:
            return {"cache_enabled": False}
        
        # Get pre-optimization stats
        pre_stats = self.cache.get_stats()
        
        # Trigger cleanup in filesystem cache
        if hasattr(self.cache, 'fs_cache') and self.cache.fs_cache:
            self.cache.fs_cache._cleanup_expired()
            self.cache.fs_cache._cleanup_if_needed()
        
        # Get post-optimization stats
        post_stats = self.cache.get_stats()
        
        self.logger.info("Cache optimization completed")
        
        return {
            "cache_enabled": True,
            "pre_optimization": pre_stats,
            "post_optimization": post_stats,
        }
    
    def benchmark_performance(self, network_path: Path, target: FPGATarget, 
                            iterations: int = 3) -> Dict[str, Any]:
        """Benchmark compilation performance."""
        self.logger.info("Starting performance benchmark",
                        network=str(network_path),
                        target=target.value,
                        iterations=iterations)
        
        results = []
        
        for i in range(iterations):
            # Clear cache for fair comparison
            if i == 0 and self.cache:
                self.clear_cache()
            
            start_time = datetime.utcnow()
            
            result = self.compile(
                network_path,
                target,
                Path(f"./benchmark_output_{i}"),
                ScalableCompilationConfig(
                    enable_caching=True,
                    enable_concurrency=False  # Single-threaded for consistent benchmarking
                )
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            iteration_result = {
                "iteration": i,
                "duration_seconds": duration,
                "success": result.success,
                "cache_hit": i > 0,  # First iteration is always cache miss
                "neurons": result.resource_estimate.neurons,
                "synapses": result.resource_estimate.synapses,
            }
            
            results.append(iteration_result)
            
            self.logger.info("Benchmark iteration completed",
                           iteration=i,
                           duration=duration,
                           cache_hit=iteration_result["cache_hit"])
        
        # Calculate statistics
        successful_runs = [r for r in results if r["success"]]
        
        if successful_runs:
            avg_duration = sum(r["duration_seconds"] for r in successful_runs) / len(successful_runs)
            
            # Compare first run (no cache) vs subsequent runs (with cache)
            first_run_duration = successful_runs[0]["duration_seconds"]
            cached_runs = [r for r in successful_runs[1:] if r["cache_hit"]]
            
            if cached_runs:
                avg_cached_duration = sum(r["duration_seconds"] for r in cached_runs) / len(cached_runs)
                speedup = first_run_duration / avg_cached_duration
            else:
                avg_cached_duration = None
                speedup = None
        else:
            avg_duration = None
            first_run_duration = None
            avg_cached_duration = None
            speedup = None
        
        benchmark_stats = {
            "network": str(network_path),
            "target": target.value,
            "iterations": iterations,
            "successful_runs": len(successful_runs),
            "average_duration_seconds": avg_duration,
            "first_run_duration_seconds": first_run_duration,
            "average_cached_duration_seconds": avg_cached_duration,
            "cache_speedup_factor": speedup,
            "detailed_results": results,
        }
        
        self.logger.info("Performance benchmark completed", **benchmark_stats)
        
        return benchmark_stats
    
    def shutdown(self) -> None:
        """Gracefully shutdown the scalable compiler."""
        self.logger.info("Shutting down scalable compiler")
        
        if self.load_balancer:
            self.load_balancer.shutdown()
        
        if self.concurrent_compiler:
            self.concurrent_compiler.shutdown()
        
        if self.compiler_pool:
            self.compiler_pool.shutdown()
        
        self.logger.info("Scalable compiler shutdown complete")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup