"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_network_small():
    """Return a small test network configuration."""
    return {
        "neurons": 100,
        "layers": [
            {"type": "input", "size": 28},
            {"type": "hidden", "size": 64, "neuron_model": "LIF"},
            {"type": "output", "size": 10}
        ],
        "connectivity": "sparse_random",
        "sparsity": 0.1
    }


@pytest.fixture
def sample_network_large():
    """Return a larger test network configuration."""
    return {
        "neurons": 10000,
        "layers": [
            {"type": "input", "size": 784},
            {"type": "hidden", "size": 1000, "neuron_model": "LIF"},
            {"type": "hidden", "size": 500, "neuron_model": "LIF"},
            {"type": "output", "size": 10}
        ],
        "connectivity": "sparse_random",
        "sparsity": 0.05
    }


@pytest.fixture(params=["artix7_35t", "artix7_100t", "cyclone5_gx"])
def fpga_target(request):
    """Parameterized fixture for different FPGA targets."""
    from spiking_fpga.core import FPGATarget
    return FPGATarget(request.param)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require FPGA hardware"
    )
    config.addinivalue_line(
        "markers", "vivado: marks tests requiring Xilinx Vivado"
    )
    config.addinivalue_line(
        "markers", "quartus: marks tests requiring Intel Quartus"
    )