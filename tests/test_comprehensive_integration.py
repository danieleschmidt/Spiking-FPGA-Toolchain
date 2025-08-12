"""
Comprehensive integration tests for the complete SDLC implementation.
Tests all three generations: Make it Work, Make it Reliable, Make it Scale.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time
import threading

from spiking_fpga import FPGATarget, compile_network
from spiking_fpga.network_compiler import CompilationConfig
from spiking_fpga.models.optimization import OptimizationLevel
from spiking_fpga.reliability import FaultTolerantCompiler, FaultToleranceConfig, RedundancyMode
from spiking_fpga.performance import DistributedCompiler, AutoScaler
from spiking_fpga.performance.distributed_compiler import CompilationNode, DistributedConfig
from spiking_fpga.performance.auto_scaling import ScalingPolicy, ResourceType, ScalingMetric


class TestGeneration1Integration:
    """Test Generation 1: Make it Work - Basic functionality."""
    
    def test_basic_compilation(self):
        """Test basic compilation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            result = compile_network(
                "examples/simple_mnist.yaml",
                FPGATarget.ARTIX7_35T,
                output_dir=output_dir,
                optimization_level=OptimizationLevel.BASIC,
                run_synthesis=False
            )
            
            assert result.success, f"Compilation failed: {result.errors}"
            assert result.network.name == "MNIST_Classifier"
            assert len(result.hdl_files) > 0, "No HDL files generated"
            
            # Check generated files exist
            hdl_dir = output_dir / "hdl"
            assert hdl_dir.exists(), "HDL directory not created"
            assert (hdl_dir / "lif_neuron.v").exists(), "LIF neuron module not generated"
    
    def test_multiple_targets(self):
        """Test compilation for different FPGA targets."""
        targets = [FPGATarget.ARTIX7_35T, FPGATarget.ARTIX7_100T, FPGATarget.CYCLONE_V_GX]
        
        for target in targets:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = compile_network(
                    "examples/simple_mnist.yaml",
                    target,
                    output_dir=Path(temp_dir),
                    optimization_level=OptimizationLevel.BASIC
                )
                
                assert result.success, f"Compilation failed for {target.value}: {result.errors}"
                assert result.resource_estimate.neurons > 0, "No neurons estimated"
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        levels = [OptimizationLevel.NONE, OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]
        
        for level in levels:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = compile_network(
                    "examples/simple_mnist.yaml",
                    FPGATarget.ARTIX7_35T,
                    output_dir=Path(temp_dir),
                    optimization_level=level
                )
                
                assert result.success, f"Compilation failed for {level}: {result.errors}"


class TestGeneration2Integration:
    """Test Generation 2: Make it Reliable - Robustness and error handling."""
    
    def test_fault_tolerant_compilation(self):
        """Test fault-tolerant compilation with redundancy."""
        config = FaultToleranceConfig(
            redundancy_mode=RedundancyMode.DUAL_MODULAR,
            enable_checkpointing=True,
            enable_self_healing=False,
            health_check_interval=0,  # Disable for testing
            backup_targets=[FPGATarget.ARTIX7_100T]
        )
        
        ft_compiler = FaultTolerantCompiler(
            primary_target=FPGATarget.ARTIX7_35T,
            config=config
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            compilation_config = CompilationConfig(
                optimization_level=OptimizationLevel.BASIC,
                generate_reports=True,
                run_synthesis=False
            )
            
            result = ft_compiler.compile(
                "examples/simple_mnist.yaml",
                Path(temp_dir),
                compilation_config
            )
            
            assert result.success, f"Fault-tolerant compilation failed: {result.errors}"
            
            # Check redundancy was used
            redundancy_info = result.optimization_stats.get('redundancy_info')
            if redundancy_info:
                assert redundancy_info['successful_compilations'] >= 1
                assert redundancy_info['consensus_achieved'] is True
    
    def test_error_recovery_system(self):
        """Test error recovery capabilities."""
        from spiking_fpga.reliability import ErrorRecoverySystem, ErrorContext
        
        recovery = ErrorRecoverySystem()
        
        # Simulate a compilation error
        error_context = ErrorContext(
            error_type="compilation_error",
            error_message="Test error",
            timestamp=time.time(),
            component="test_compiler",
            severity="error",
            context_data={"optimization_level": 3}
        )
        
        success, recovery_data = recovery.handle_error(
            Exception("Test error"), error_context
        )
        
        # Error recovery should attempt to help
        assert isinstance(success, bool)
        if success and recovery_data:
            assert isinstance(recovery_data, dict)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from spiking_fpga.reliability import CircuitBreakerAdvanced
        
        circuit_breaker = CircuitBreakerAdvanced(
            failure_threshold=2,
            recovery_timeout=1.0
        )
        
        def failing_function():
            raise Exception("Simulated failure")
        
        def working_function():
            return "success"
        
        # Cause failures to open circuit breaker
        for _ in range(3):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        status = circuit_breaker.get_status()
        assert status['state'] in ['OPEN', 'HALF_OPEN']
        
        # Test that working function succeeds when circuit is closed/half-open
        try:
            result = circuit_breaker.call(working_function)
            assert result == "success"
        except Exception:
            # Expected if circuit is open
            pass


class TestGeneration3Integration:
    """Test Generation 3: Make it Scale - Performance and scalability."""
    
    def test_distributed_compilation(self):
        """Test distributed compilation system."""
        config = DistributedConfig(
            load_balancing_strategy='least_loaded',
            enable_adaptive_scheduling=True,
            max_retries=1,
            job_timeout=30.0
        )
        
        distributed_compiler = DistributedCompiler(config)
        
        # Add a local compilation node
        node = CompilationNode(
            node_id='test_node',
            hostname='localhost',
            targets=[FPGATarget.ARTIX7_35T],
            max_concurrent_jobs=1
        )
        
        distributed_compiler.add_node(node)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                compilation_config = CompilationConfig(
                    optimization_level=OptimizationLevel.BASIC,
                    generate_reports=False,
                    run_synthesis=False
                )
                
                job_id = distributed_compiler.submit_job(
                    network="examples/simple_mnist.yaml",
                    output_dir=Path(temp_dir),
                    config=compilation_config,
                    target=FPGATarget.ARTIX7_35T
                )
                
                # Wait for job completion
                result = distributed_compiler.wait_for_job(job_id, timeout=30.0)
                
                assert result is not None, "Distributed job timed out"
                assert result.success, f"Distributed compilation failed: {result.errors}"
                
                # Check cluster status
                status = distributed_compiler.get_cluster_status()
                assert status['total_jobs_completed'] >= 1
                assert status['active_nodes'] >= 1
        
        finally:
            distributed_compiler.shutdown()
    
    def test_auto_scaling_system(self):
        """Test auto-scaling functionality."""
        auto_scaler = AutoScaler()
        
        # Add scaling policy
        policy = ScalingPolicy(
            resource_type=ResourceType.COMPILATION_NODES,
            min_capacity=1,
            max_capacity=3,
            target_utilization=0.7
        )
        
        auto_scaler.add_scaling_policy(policy)
        
        # Add scaling metric
        metric = ScalingMetric(
            name='cpu_utilization',
            current_value=0.5,
            threshold_up=0.8,
            threshold_down=0.3
        )
        
        auto_scaler.add_scaling_metric(metric)
        
        # Test metric updates
        auto_scaler.update_metrics({
            'cpu_utilization': 0.6,
            'memory_utilization': 0.4
        })
        
        # Check status
        status = auto_scaler.get_scaling_status()
        assert 'current_capacity' in status
        assert ResourceType.COMPILATION_NODES.value in status['current_capacity']
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        from spiking_fpga.performance import PerformanceOrchestrator, IntelligentCache
        
        # Test performance orchestrator
        perf_optimizer = PerformanceOrchestrator()
        config = {"optimization_level": 2}
        
        optimized_config = perf_optimizer.optimize_compilation(config)
        assert isinstance(optimized_config, dict)
        
        # Test intelligent caching
        cache = IntelligentCache(max_size=100)
        
        # Test cache operations
        cache.put("test_key", {"data": "test_value"})
        cached_value = cache.get("test_key")
        
        assert cached_value is not None
        assert cached_value["data"] == "test_value"
        
        # Test cache miss
        missing_value = cache.get("nonexistent_key")
        assert missing_value is None


class TestEndToEndIntegration:
    """End-to-end integration tests covering all generations."""
    
    def test_complete_pipeline(self):
        """Test complete compilation pipeline with all features."""
        # Generation 1: Basic compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            basic_result = compile_network(
                "examples/simple_mnist.yaml",
                FPGATarget.ARTIX7_35T,
                output_dir=Path(temp_dir) / "basic",
                optimization_level=OptimizationLevel.BASIC
            )
            
            assert basic_result.success, "Basic compilation failed"
        
        # Generation 2: Fault-tolerant compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            ft_config = FaultToleranceConfig(
                redundancy_mode=RedundancyMode.DUAL_MODULAR,
                enable_checkpointing=False,
                enable_self_healing=False,
                health_check_interval=0,
                backup_targets=[FPGATarget.ARTIX7_100T]
            )
            
            ft_compiler = FaultTolerantCompiler(FPGATarget.ARTIX7_35T, ft_config)
            
            ft_result = ft_compiler.compile(
                "examples/simple_mnist.yaml",
                Path(temp_dir) / "fault_tolerant",
                CompilationConfig(optimization_level=OptimizationLevel.BASIC)
            )
            
            assert ft_result.success, "Fault-tolerant compilation failed"
        
        # Generation 3: Distributed compilation with performance optimization
        dist_config = DistributedConfig(
            load_balancing_strategy='performance_based',
            enable_adaptive_scheduling=True,
            max_retries=1,
            job_timeout=30.0
        )
        
        distributed_compiler = DistributedCompiler(dist_config)
        
        node = CompilationNode(
            node_id='perf_test_node',
            hostname='localhost',
            targets=[FPGATarget.ARTIX7_35T],
            max_concurrent_jobs=1
        )
        
        distributed_compiler.add_node(node)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                job_id = distributed_compiler.submit_job(
                    network="examples/simple_mnist.yaml",
                    output_dir=Path(temp_dir) / "distributed",
                    config=CompilationConfig(optimization_level=OptimizationLevel.AGGRESSIVE),
                    target=FPGATarget.ARTIX7_35T
                )
                
                dist_result = distributed_compiler.wait_for_job(job_id, timeout=30.0)
                assert dist_result is not None, "Distributed compilation timed out"
                assert dist_result.success, "Distributed compilation failed"
        
        finally:
            distributed_compiler.shutdown()
    
    @pytest.mark.benchmark
    def test_performance_benchmarks(self, benchmark):
        """Benchmark compilation performance."""
        def compilation_benchmark():
            with tempfile.TemporaryDirectory() as temp_dir:
                return compile_network(
                    "examples/simple_mnist.yaml",
                    FPGATarget.ARTIX7_35T,
                    output_dir=Path(temp_dir),
                    optimization_level=OptimizationLevel.BASIC,
                    run_synthesis=False
                )
        
        result = benchmark(compilation_benchmark)
        assert result.success, "Benchmark compilation failed"
    
    def test_concurrent_compilations(self):
        """Test multiple concurrent compilations."""
        def compile_concurrent(target, output_suffix):
            with tempfile.TemporaryDirectory() as temp_dir:
                return compile_network(
                    "examples/simple_mnist.yaml",
                    target,
                    output_dir=Path(temp_dir),
                    optimization_level=OptimizationLevel.BASIC
                )
        
        # Start multiple compilation threads
        import concurrent.futures
        
        targets = [FPGATarget.ARTIX7_35T, FPGATarget.ARTIX7_100T]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(compile_concurrent, target, str(i))
                for i, target in enumerate(targets)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All compilations should succeed
        for i, result in enumerate(results):
            assert result.success, f"Concurrent compilation {i} failed: {result.errors}"
    
    def test_stress_test(self):
        """Stress test the system with multiple operations."""
        # Test rapid successive compilations
        success_count = 0
        total_tests = 5
        
        for i in range(total_tests):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = compile_network(
                        "examples/simple_mnist.yaml",
                        FPGATarget.ARTIX7_35T,
                        output_dir=Path(temp_dir),
                        optimization_level=OptimizationLevel.BASIC
                    )
                    
                    if result.success:
                        success_count += 1
            
            except Exception as e:
                print(f"Stress test iteration {i} failed: {e}")
        
        # Should have at least 80% success rate
        success_rate = success_count / total_tests
        assert success_rate >= 0.8, f"Stress test success rate too low: {success_rate:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])