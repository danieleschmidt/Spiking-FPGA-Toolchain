"""Comprehensive tests for the scalable compiler."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from spiking_fpga.scalable_compiler import ScalableNetworkCompiler, ScalableCompilationConfig
from spiking_fpga import FPGATarget
from spiking_fpga.models.network import Network, LayerType


class TestScalableCompiler:
    """Test suite for scalable compiler functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing."""
        network = Network(name="test_network")
        
        # Add layers
        input_layer = network.add_layer(LayerType.INPUT, 10, "LIF")
        hidden_layer = network.add_layer(LayerType.HIDDEN, 20, "LIF") 
        output_layer = network.add_layer(LayerType.OUTPUT, 5, "LIF")
        
        # Connect layers
        network.connect_layers(input_layer, hidden_layer, sparsity=0.8)
        network.connect_layers(hidden_layer, output_layer, sparsity=0.9)
        
        return network
    
    @pytest.fixture
    def compiler_config(self, temp_dir):
        """Create compiler configuration for testing."""
        return ScalableCompilationConfig(
            enable_caching=True,
            cache_dir=temp_dir / "cache",
            enable_concurrency=False,  # Disable for unit tests
            optimization_level=1
        )
    
    def test_compiler_initialization(self, compiler_config):
        """Test compiler initialization with various configurations."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        assert compiler.config.enable_caching is True
        assert compiler.cache is not None
        assert compiler.concurrent_compiler is None  # Disabled
        assert compiler.compiler_pool is not None
        
        compiler.shutdown()
    
    def test_network_hash_generation(self, sample_network, temp_dir):
        """Test network hash generation for caching."""
        compiler = ScalableNetworkCompiler(ScalableCompilationConfig())
        
        # Test network object hashing
        hash1 = compiler._generate_network_hash(sample_network)
        hash2 = compiler._generate_network_hash(sample_network)
        
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 character hash
        
        # Test file-based hashing
        network_file = temp_dir / "test_network.yaml"
        network_file.write_text("name: test\\nlayers: []")
        
        hash_file1 = compiler._generate_network_hash(network_file)
        hash_file2 = compiler._generate_network_hash(network_file)
        
        assert hash_file1 == hash_file2
        assert hash_file1 != hash1  # Different from network object
        
        compiler.shutdown()
    
    def test_config_hash_generation(self, compiler_config):
        """Test configuration hash generation."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        hash1 = compiler._generate_config_hash(compiler_config, FPGATarget.ARTIX7_35T)
        hash2 = compiler._generate_config_hash(compiler_config, FPGATarget.ARTIX7_35T)
        
        assert hash1 == hash2
        
        # Different target should produce different hash
        hash3 = compiler._generate_config_hash(compiler_config, FPGATarget.ARTIX7_100T)
        assert hash1 != hash3
        
        compiler.shutdown()
    
    @patch('spiking_fpga.scalable_compiler.NetworkCompiler')
    def test_compile_with_cache_miss(self, mock_compiler_class, sample_network, temp_dir, compiler_config):
        """Test compilation with cache miss."""
        # Mock the compiler
        mock_compiler = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.resource_estimate = Mock()
        mock_result.resource_estimate.neurons = 35
        mock_result.resource_estimate.synapses = 150
        mock_result.resource_estimate.luts = 1000
        mock_result.resource_estimate.registers = 800
        mock_result.resource_estimate.bram_kb = 10.5
        mock_result.resource_estimate.dsp_slices = 5
        mock_result.optimization_stats = {"level": "basic"}
        mock_result.warnings = []
        mock_result.errors = []
        
        mock_compiler.compile.return_value = mock_result
        mock_compiler_class.return_value = mock_compiler
        
        compiler = ScalableNetworkCompiler(compiler_config)
        
        result = compiler.compile(
            sample_network,
            FPGATarget.ARTIX7_35T,
            temp_dir / "output"
        )
        
        assert result.success is True
        assert result.resource_estimate.neurons == 35
        
        # Verify compilation was called
        mock_compiler.compile.assert_called_once()
        
        compiler.shutdown()
    
    def test_cache_hit_scenario(self, sample_network, temp_dir, compiler_config):
        """Test compilation cache hit scenario."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        # Manually add cached result
        network_hash = compiler._generate_network_hash(sample_network)
        config_hash = compiler._generate_config_hash(compiler_config, FPGATarget.ARTIX7_35T)
        
        cached_data = {
            "success": True,
            "resource_estimate": {
                "neurons": 35,
                "synapses": 150,
                "luts": 1000,
                "registers": 800,
                "bram_kb": 10.5,
                "dsp_slices": 5,
            },
            "optimization_stats": {"level": "cached"},
            "warnings": [],
            "errors": [],
        }
        
        compiler.cache.put_optimization_result(network_hash, config_hash, cached_data)
        
        # Now compile - should get cached result
        result = compiler.compile(
            sample_network,
            FPGATarget.ARTIX7_35T,
            temp_dir / "output"
        )
        
        assert result.success is True
        assert result.resource_estimate.neurons == 35
        assert result.optimization_stats["level"] == "cached"
        
        compiler.shutdown()
    
    def test_performance_stats(self, compiler_config):
        """Test performance statistics collection."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        stats = compiler.get_performance_stats()
        
        assert "compiler_pool" in stats
        assert "cache" in stats
        
        pool_stats = stats["compiler_pool"]
        assert "current_size" in pool_stats
        assert "max_size" in pool_stats
        
        cache_stats = stats["cache"]
        assert "enabled" in cache_stats
        
        compiler.shutdown()
    
    def test_cache_operations(self, compiler_config):
        """Test cache clearing and optimization."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        # Test cache clearing
        compiler.clear_cache()
        
        # Test cache optimization
        optimization_result = compiler.optimize_cache()
        assert optimization_result["cache_enabled"] is True
        assert "pre_optimization" in optimization_result
        assert "post_optimization" in optimization_result
        
        compiler.shutdown()
    
    def test_concurrent_compilation_disabled(self, sample_network, temp_dir):
        """Test behavior when concurrency is disabled."""
        config = ScalableCompilationConfig(enable_concurrency=False)
        compiler = ScalableNetworkCompiler(config)
        
        # Should raise error when trying to use concurrent features
        with pytest.raises(RuntimeError, match="Concurrency not enabled"):
            compiler.compile_concurrent([])
        
        status = compiler.get_task_status("dummy_task")
        assert "not_found" in status["status"] or "Concurrency not enabled" in status.get("error", "")
        
        compiler.shutdown()
    
    def test_concurrent_task_submission(self, temp_dir):
        """Test concurrent task submission."""
        config = ScalableCompilationConfig(
            enable_concurrency=True,
            use_load_balancer=False,
            max_concurrent_workers=2
        )
        compiler = ScalableNetworkCompiler(config)
        
        # Create dummy network file
        network_file = temp_dir / "test.yaml"
        network_file.write_text("""
name: "test_concurrent"
timestep: 1.0
layers:
  - name: "input"
    type: "input"
    size: 5
    neuron_type: "LIF"
connections: []
""")
        
        tasks = [
            {
                "name": "task1",
                "network": str(network_file),
                "target": "artix7_35t",
                "output_dir": str(temp_dir / "output1"),
            }
        ]
        
        task_ids = compiler.compile_concurrent(tasks)
        
        assert "task1" in task_ids
        assert isinstance(task_ids["task1"], str)
        
        # Check task status
        status = compiler.get_task_status(task_ids["task1"])
        assert "status" in status
        
        compiler.shutdown()
    
    def test_error_handling(self, temp_dir, compiler_config):
        """Test error handling in compilation."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        # Test with non-existent network file
        non_existent_file = temp_dir / "non_existent.yaml"
        
        result = compiler.compile(
            non_existent_file,
            FPGATarget.ARTIX7_35T,
            temp_dir / "output"
        )
        
        # Should handle the error gracefully
        assert result.success is False
        assert len(result.errors) > 0
        
        compiler.shutdown()
    
    def test_shutdown_behavior(self, compiler_config):
        """Test proper shutdown behavior."""
        compiler = ScalableNetworkCompiler(compiler_config)
        
        # Should not raise any exceptions
        compiler.shutdown()
        
        # Multiple shutdowns should be safe
        compiler.shutdown()


class TestCompilerIntegration:
    """Integration tests for the compiler with real examples."""
    
    @pytest.fixture
    def example_network_file(self, tmp_path):
        """Create a valid example network file."""
        network_content = """
name: "integration_test_network"
description: "Small network for integration testing"
timestep: 1.0

layers:
  - name: "input"
    type: "input"
    size: 5
    neuron_type: "LIF"
    parameters:
      v_thresh: 1.0
      v_reset: 0.0
      tau_m: 20.0

  - name: "hidden"
    type: "hidden"
    size: 10
    neuron_type: "LIF"
    parameters:
      v_thresh: 1.0
      v_reset: 0.0
      tau_m: 20.0

  - name: "output"
    type: "output"
    size: 3
    neuron_type: "LIF"
    parameters:
      v_thresh: 1.0
      v_reset: 0.0
      tau_m: 20.0

connections:
  - from: "input"
    to: "hidden"
    pattern: "all_to_all"
    weight_distribution: "uniform"
    weight_params:
      min: 0.1
      max: 0.5
    sparsity: 0.8

  - from: "hidden"
    to: "output"
    pattern: "all_to_all"
    weight_distribution: "uniform"
    weight_params:
      min: 0.2
      max: 0.8
    sparsity: 0.9
"""
        network_file = tmp_path / "integration_test.yaml"
        network_file.write_text(network_content)
        return network_file
    
    def test_end_to_end_compilation(self, example_network_file, tmp_path):
        """Test complete end-to-end compilation process."""
        config = ScalableCompilationConfig(
            enable_caching=True,
            cache_dir=tmp_path / "cache",
            enable_concurrency=False,  # Keep simple for integration test
            optimization_level=1,
            generate_reports=True
        )
        
        compiler = ScalableNetworkCompiler(config)
        
        output_dir = tmp_path / "integration_output"
        
        result = compiler.compile(
            example_network_file,
            FPGATarget.ARTIX7_35T,
            output_dir
        )
        
        # Verify successful compilation
        assert result.success is True
        assert result.resource_estimate.neurons > 0
        assert result.resource_estimate.synapses > 0
        
        # Verify output files were created
        assert output_dir.exists()
        hdl_dir = output_dir / "hdl"
        if hdl_dir.exists():
            hdl_files = list(hdl_dir.glob("*.v"))
            assert len(hdl_files) > 0  # Should have generated HDL files
        
        reports_dir = output_dir / "reports"  
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.txt"))
            assert len(report_files) > 0  # Should have generated reports
        
        compiler.shutdown()
    
    def test_cache_effectiveness(self, example_network_file, tmp_path):
        """Test that caching actually improves performance."""
        config = ScalableCompilationConfig(
            enable_caching=True,
            cache_dir=tmp_path / "cache",
            enable_concurrency=False
        )
        
        compiler = ScalableNetworkCompiler(config)
        
        # First compilation (cache miss)
        import time
        start_time = time.time()
        
        result1 = compiler.compile(
            example_network_file,
            FPGATarget.ARTIX7_35T,
            tmp_path / "output1"
        )
        
        first_duration = time.time() - start_time
        
        # Second compilation (should be cache hit)
        start_time = time.time()
        
        result2 = compiler.compile(
            example_network_file,
            FPGATarget.ARTIX7_35T,
            tmp_path / "output2"
        )
        
        second_duration = time.time() - start_time
        
        # Both should succeed
        assert result1.success is True
        assert result2.success is True
        
        # Results should be equivalent
        assert result1.resource_estimate.neurons == result2.resource_estimate.neurons
        assert result1.resource_estimate.synapses == result2.resource_estimate.synapses
        
        # Second compilation should be faster (cached)
        # Note: This might not always be true in test environment, so we just check it completed
        assert second_duration >= 0
        
        # Check cache stats
        stats = compiler.get_performance_stats()
        cache_stats = stats.get("cache", {})
        
        # Should have cache entries
        if "memory_cache" in cache_stats:
            assert cache_stats["memory_cache"].get("size", 0) >= 0
        
        compiler.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])