"""Unit tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from spiking_fpga.cli import main


class TestCLI:
    """Test command-line interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_main_help(self):
        """Test main command help output."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "Spiking-FPGA-Toolchain" in result.output
        assert "compile" in result.output
        assert "resources" in result.output
    
    def test_compile_help(self):
        """Test compile subcommand help."""
        result = self.runner.invoke(main, ['compile', '--help'])
        assert result.exit_code == 0
        assert "Compile a spiking neural network" in result.output
        assert "--target" in result.output
        assert "--output" in result.output
    
    def test_resources_command(self):
        """Test resources command output."""
        result = self.runner.invoke(main, ['resources'])
        assert result.exit_code == 0
        assert "Supported FPGA targets" in result.output
        assert "artix7_35t" in result.output
        assert "cyclone5_gx" in result.output
    
    def test_resources_specific_target(self):
        """Test resources command for specific target."""
        result = self.runner.invoke(main, ['resources', '--target', 'artix7_35t'])
        assert result.exit_code == 0
        assert "Resources for artix7_35t" in result.output
        assert "logic_cells" in result.output
        assert "max_neurons" in result.output
    
    def test_validate_command(self):
        """Test validate command."""
        result = self.runner.invoke(main, ['validate'])
        assert result.exit_code == 0
        assert "Validating environment" in result.output
        assert "Python dependencies OK" in result.output
    
    def test_compile_placeholder(self):
        """Test compile command placeholder behavior."""
        with self.runner.isolated_filesystem():
            # Create a dummy network file
            with open('test_network.yaml', 'w') as f:
                f.write('neurons: 100\n')
            
            result = self.runner.invoke(main, [
                'compile', 'test_network.yaml',
                '--target', 'artix7_35t'
            ])
            assert result.exit_code == 0
            assert "Compiling test_network.yaml" in result.output
            # The CLI now actually tries to compile, so check for compilation output
            assert ("compilation" in result.output.lower() or 
                   "validation error" in result.output.lower())
    
    def test_invalid_target(self):
        """Test compile with invalid target."""
        with self.runner.isolated_filesystem():
            with open('test_network.yaml', 'w') as f:
                f.write('neurons: 100\n')
            
            result = self.runner.invoke(main, [
                'compile', 'test_network.yaml',
                '--target', 'invalid_target'
            ])
            assert result.exit_code != 0