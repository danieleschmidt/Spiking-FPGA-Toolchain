"""
Integration tests for the complete compilation pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.spiking_fpga.services.network_compiler import NetworkCompiler, CompilationConfig
from src.spiking_fpga.parsers.yaml_parser import YAMLNetworkParser
from src.spiking_fpga.models.fpga import FPGATarget


@pytest.mark.integration
class TestCompilationPipeline:
    """Test complete compilation from YAML to HDL."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_simple_network_compilation(self, simple_lif_network):
        """Test compilation of a simple network."""
        config = CompilationConfig(
            optimization_level=1,
            target_frequency_mhz=100.0,
            output_directory=str(self.output_dir)
        )
        
        compiler = NetworkCompiler(config)
        result = compiler.compile_network(simple_lif_network, FPGATarget.ARTIX7_35T)
        
        assert result.success
        assert result.resource_utilization is not None
        assert result.compilation_time_s > 0
        assert len(result.hdl_files) > 0
        
        # Check that files were actually created
        for hdl_file in result.hdl_files:
            assert Path(hdl_file).exists()
    
    def test_yaml_to_hdl_pipeline(self, sample_networks):
        """Test complete pipeline from YAML to HDL."""
        # Create temporary YAML file
        yaml_file = Path(self.temp_dir) / "test_network.yaml"
        
        parser = YAMLNetworkParser()
        
        # Convert sample network dict to actual network object
        network_dict = sample_networks['mnist_classifier']
        network = parser.parse_config(network_dict)
        
        # Write to YAML file
        parser.write_config(network, str(yaml_file))
        
        # Parse it back
        parsed_network = parser.parse_file(str(yaml_file))
        
        # Compile the parsed network
        config = CompilationConfig(output_directory=str(self.output_dir))
        compiler = NetworkCompiler(config)
        result = compiler.compile_network(parsed_network, FPGATarget.ARTIX7_35T)
        
        assert result.success
        assert result.network.name == network_dict['name']
    
    def test_compilation_with_caching(self, simple_lif_network):
        """Test compilation with caching enabled."""
        config = CompilationConfig(
            enable_caching=True,
            output_directory=str(self.output_dir)
        )
        
        compiler = NetworkCompiler(config)
        
        # First compilation
        result1 = compiler.compile_network(simple_lif_network, FPGATarget.ARTIX7_35T)
        assert result1.success
        time1 = result1.compilation_time_s
        
        # Second compilation (should be faster due to caching)
        result2 = compiler.compile_network(simple_lif_network, FPGATarget.ARTIX7_35T)
        assert result2.success
        
        # Both should produce similar results
        assert abs(result1.resource_utilization.luts_used - result2.resource_utilization.luts_used) < 10


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for compilation pipeline."""
    
    def test_large_network_compilation_time(self, complex_network):
        """Benchmark compilation time for complex networks."""
        config = CompilationConfig(optimization_level=2)
        compiler = NetworkCompiler(config)
        
        result = compiler.compile_network(complex_network, FPGATarget.ARTIX7_100T)
        
        assert result.success
        assert result.compilation_time_s < 60.0  # Should complete within 1 minute
        print(f"Complex network compilation time: {result.compilation_time_s:.2f}s")
    
    def test_optimization_level_performance(self, simple_lif_network):
        """Test performance impact of different optimization levels."""
        results = {}
        
        for opt_level in [0, 1, 2, 3]:
            config = CompilationConfig(optimization_level=opt_level)
            compiler = NetworkCompiler(config)
            
            result = compiler.compile_network(simple_lif_network, FPGATarget.ARTIX7_35T)
            assert result.success
            
            results[opt_level] = {
                'time': result.compilation_time_s,
                'luts': result.resource_utilization.luts_used,
                'bram': result.resource_utilization.bram_used
            }
        
        # Higher optimization should generally use fewer resources
        assert results[2]['luts'] <= results[0]['luts']
        
        print("Optimization level performance:")
        for level, metrics in results.items():
            print(f"  Level {level}: {metrics['time']:.2f}s, {metrics['luts']} LUTs")


@pytest.mark.hardware
class TestHardwareIntegration:
    """Hardware-in-the-loop tests (requires actual FPGA board)."""
    
    def test_bitstream_programming(self, simple_lif_network):
        """Test actual FPGA programming (requires hardware)."""
        pytest.skip("Hardware tests require FPGA board connection")
        
        # This would test:
        # 1. Compilation to bitstream
        # 2. Programming FPGA
        # 3. Functional verification
        # 4. Performance measurement
    
    def test_spike_injection_and_monitoring(self):
        """Test spike injection and output monitoring."""
        pytest.skip("Hardware tests require FPGA board connection")
        
        # This would test:
        # 1. Inject test spike patterns
        # 2. Monitor output spikes
        # 3. Verify timing and functionality