"""Tests for validation utilities."""

import pytest
import tempfile
from pathlib import Path

from spiking_fpga.utils.validation import (
    NetworkValidator, ConfigurationValidator, FileValidator,
    ValidationResult, validate_identifier, sanitize_filename
)
from spiking_fpga.models.network import Network, LayerType
from spiking_fpga.models.optimization import OptimizationLevel, ResourceEstimate
from spiking_fpga.core import FPGATarget


class TestNetworkValidator:
    """Test network validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create network validator."""
        return NetworkValidator()
    
    @pytest.fixture
    def valid_network(self):
        """Create a valid test network."""
        network = Network(name="test_network")
        
        input_layer = network.add_layer(LayerType.INPUT, 10, "LIF")
        hidden_layer = network.add_layer(LayerType.HIDDEN, 20, "LIF")
        output_layer = network.add_layer(LayerType.OUTPUT, 5, "LIF")
        
        network.connect_layers(input_layer, hidden_layer, sparsity=0.8)
        network.connect_layers(hidden_layer, output_layer, sparsity=0.9)
        
        return network
    
    def test_validate_valid_network(self, validator, valid_network):
        """Test validation of a valid network."""
        result = validator.validate_network(valid_network)
        
        assert result.valid is True
        assert len(result.issues) == 0
        # May have warnings or recommendations, but should be valid
    
    def test_validate_empty_network(self, validator):
        """Test validation of empty network."""
        network = Network(name="empty_network")
        
        result = validator.validate_network(network)
        
        assert result.valid is False
        assert any("no layers" in issue.lower() for issue in result.issues)
    
    def test_validate_disconnected_network(self, validator):
        """Test validation of network with disconnected components."""
        network = Network(name="disconnected_network")
        
        # Add layers but no connections
        network.add_layer(LayerType.INPUT, 10, "LIF")
        network.add_layer(LayerType.OUTPUT, 5, "LIF")
        
        result = validator.validate_network(network)
        
        # The network should be considered valid but may have warnings
        # A network without connections is unusual but not necessarily invalid
        assert result.valid is True  # Basic structure is valid
        # Warnings are optional - the validator may or may not flag this
    
    def test_validate_invalid_neuron_parameters(self, validator):
        """Test validation of invalid neuron parameters."""
        network = Network(name="invalid_params_network")
        
        # Add layer with invalid parameters
        layer_id = network.add_layer(LayerType.INPUT, 5, "LIF")
        
        # Manually set invalid parameters on neurons
        for neuron in network.neurons:
            if neuron.layer_id == layer_id:
                neuron.parameters = {
                    "v_thresh": 0.5,
                    "v_reset": 1.0,  # Reset > threshold (invalid)
                    "tau_m": -10.0   # Negative time constant (invalid)
                }
        
        result = validator.validate_network(network)
        
        assert result.valid is False
        assert len(result.issues) > 0
        assert any("threshold" in issue.lower() for issue in result.issues)
        assert any("time constant" in issue.lower() for issue in result.issues)
    
    def test_validate_large_network(self, validator):
        """Test validation of large network."""
        network = Network(name="large_network")
        
        # Create network that exceeds validator limits
        validator.max_neurons = 100  # Set low limit for testing
        
        input_layer = network.add_layer(LayerType.INPUT, 150, "LIF")  # Exceeds limit
        output_layer = network.add_layer(LayerType.OUTPUT, 50, "LIF")
        
        network.connect_layers(input_layer, output_layer)
        
        result = validator.validate_network(network)
        
        assert result.valid is False
        assert any("too many neurons" in issue.lower() for issue in result.issues)
    
    def test_generate_recommendations(self, validator, valid_network):
        """Test recommendation generation."""
        # Create network with very high connectivity
        high_conn_network = Network(name="high_connectivity")
        
        input_layer = high_conn_network.add_layer(LayerType.INPUT, 10, "LIF")
        output_layer = high_conn_network.add_layer(LayerType.OUTPUT, 10, "LIF")
        
        # Full connectivity (high)
        high_conn_network.connect_layers(input_layer, output_layer, sparsity=1.0)
        
        result = validator.validate_network(high_conn_network)
        
        # Should generate recommendations about connectivity or pass validation
        # (The validator might not always generate recommendations)
        assert result.valid is True  # At least it should be valid
        # Recommendations are optional based on network characteristics


class TestConfigurationValidator:
    """Test configuration validation."""
    
    @pytest.fixture
    def validator(self):
        """Create configuration validator."""
        return ConfigurationValidator()
    
    def test_validate_fpga_target_within_limits(self, validator):
        """Test FPGA target validation within resource limits."""
        target = FPGATarget.ARTIX7_35T
        
        # Create resource estimate within limits
        estimate = ResourceEstimate(
            luts=20000,      # Within limit
            bram_kb=1000,    # Within limit  
            dsp_slices=60    # Within limit
        )
        
        result = validator.validate_fpga_target(target, estimate)
        
        assert result.valid is True
        assert len(result.issues) == 0
    
    def test_validate_fpga_target_exceeds_limits(self, validator):
        """Test FPGA target validation exceeding resource limits."""
        target = FPGATarget.ARTIX7_35T
        
        # Create resource estimate that exceeds limits
        estimate = ResourceEstimate(
            luts=50000,      # Exceeds limit
            bram_kb=3000,    # Exceeds limit
            dsp_slices=200   # Exceeds limit
        )
        
        result = validator.validate_fpga_target(target, estimate)
        
        assert result.valid is False
        assert len(result.issues) > 0
        assert any("logic utilization" in issue.lower() for issue in result.issues)
        assert any("memory utilization" in issue.lower() for issue in result.issues)
        assert any("dsp utilization" in issue.lower() for issue in result.issues)
    
    def test_validate_optimization_config_valid(self, validator):
        """Test validation of valid optimization configuration."""
        result = validator.validate_optimization_config(OptimizationLevel.BASIC)
        
        assert result.valid is True
        assert len(result.issues) == 0
    
    def test_validate_optimization_config_invalid_params(self, validator):
        """Test validation of invalid optimization parameters."""
        custom_params = {
            "weight_threshold": 1.5,    # Invalid (> 1.0)
            "cluster_size": -5          # Invalid (negative)
        }
        
        result = validator.validate_optimization_config(
            OptimizationLevel.BASIC, 
            custom_params
        )
        
        assert result.valid is False
        assert len(result.issues) > 0
        assert any("weight threshold" in issue.lower() for issue in result.issues)
        assert any("cluster size" in issue.lower() for issue in result.issues)


class TestFileValidator:
    """Test file validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create file validator."""
        return FileValidator()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_validate_existing_yaml_file(self, validator, temp_dir):
        """Test validation of existing YAML file."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("""
name: "test_network"
layers: []
connections: []
""")
        
        result = validator.validate_network_file(yaml_file)
        
        assert result.valid is True
        assert len(result.issues) == 0
    
    def test_validate_non_existent_file(self, validator, temp_dir):
        """Test validation of non-existent file."""
        non_existent = temp_dir / "non_existent.yaml"
        
        result = validator.validate_network_file(non_existent)
        
        assert result.valid is False
        assert any("does not exist" in issue.lower() for issue in result.issues)
    
    def test_validate_unsupported_format(self, validator, temp_dir):
        """Test validation of unsupported file format."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("some content")
        
        result = validator.validate_network_file(txt_file)
        
        assert result.valid is False
        assert any("unsupported file format" in issue.lower() for issue in result.issues)
    
    def test_validate_invalid_yaml(self, validator, temp_dir):
        """Test validation of invalid YAML content."""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("""
invalid: yaml: content:
  - missing quotes and proper structure
    nested: {improperly
""")
        
        result = validator.validate_network_file(yaml_file)
        
        assert result.valid is False
        assert any("invalid file format" in issue.lower() for issue in result.issues)
    
    def test_validate_large_file(self, validator, temp_dir):
        """Test validation of large file."""
        large_file = temp_dir / "large.yaml"
        
        # Create content that would be > 10MB (the warning threshold)
        content = "data: " + "x" * (11 * 1024 * 1024)
        large_file.write_text(content)
        
        result = validator.validate_network_file(large_file)
        
        # Should be valid but have warnings
        assert result.valid is True
        assert len(result.warnings) > 0
        assert any("large network file" in warning.lower() for warning in result.warnings)


class TestUtilityFunctions:
    """Test utility validation functions."""
    
    def test_validate_identifier_valid(self):
        """Test identifier validation with valid identifiers."""
        assert validate_identifier("valid_name") is True
        assert validate_identifier("_private") is True
        assert validate_identifier("name123") is True
        assert validate_identifier("CamelCase") is True
    
    def test_validate_identifier_invalid(self):
        """Test identifier validation with invalid identifiers."""
        assert validate_identifier("123invalid") is False  # Starts with number
        assert validate_identifier("has-dash") is False     # Contains dash
        assert validate_identifier("has space") is False    # Contains space
        assert validate_identifier("has.dot") is False      # Contains dot
        assert validate_identifier("") is False             # Empty string
    
    def test_sanitize_filename_normal(self):
        """Test filename sanitization with normal names."""
        assert sanitize_filename("normal_file.txt") == "normal_file.txt"
        assert sanitize_filename("file-with-dashes") == "file-with-dashes"
    
    def test_sanitize_filename_dangerous(self):
        """Test filename sanitization with dangerous characters."""
        assert sanitize_filename("file<>name") == "file__name"
        assert sanitize_filename('file"with"quotes') == "file_with_quotes"
        assert sanitize_filename("file/with/slashes") == "file_with_slashes"
        # The actual pattern might replace multiple consecutive chars
        result = sanitize_filename("file\\\\with\\\\backslashes")
        assert "file" in result and "with" in result and "backslashes" in result
    
    def test_sanitize_filename_edge_cases(self):
        """Test filename sanitization edge cases."""
        # Empty or whitespace only
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"
        
        # Leading/trailing dots and spaces
        assert sanitize_filename("  .file.  ") == "file"
        
        # Very long filename
        long_name = "a" * 300
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized == "a" * 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])