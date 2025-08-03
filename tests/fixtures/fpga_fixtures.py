"""
FPGA target and resource fixtures for testing.
"""

import pytest
from src.spiking_fpga.models.fpga import FPGATarget, ResourceUtilization, TimingConstraints


@pytest.fixture
def artix7_target():
    """Artix-7 35T FPGA target for testing."""
    return FPGATarget(FPGATarget.ARTIX7_35T)


@pytest.fixture
def cyclone_v_target():
    """Cyclone V GX FPGA target for testing."""
    return FPGATarget(FPGATarget.CYCLONE_V_GX)


@pytest.fixture
def fpga_targets():
    """All supported FPGA targets."""
    return {
        'artix7_35t': FPGATarget(FPGATarget.ARTIX7_35T),
        'artix7_100t': FPGATarget(FPGATarget.ARTIX7_100T),
        'cyclone_v_gx': FPGATarget(FPGATarget.CYCLONE_V_GX),
        'cyclone_v_gt': FPGATarget(FPGATarget.CYCLONE_V_GT)
    }


@pytest.fixture
def sample_resource_utilization():
    """Sample resource utilization for testing."""
    return ResourceUtilization(
        luts_used=5000,
        flip_flops_used=8000,
        bram_used=500000,  # 500KB
        dsp_used=25,
        io_used=50
    )


@pytest.fixture
def timing_constraints():
    """Sample timing constraints for testing."""
    return TimingConstraints(
        clock_frequency_mhz=100.0,
        setup_time_ns=2.0,
        hold_time_ns=1.0,
        clock_to_output_ns=5.0,
        max_combinational_delay_ns=8.0,
        pipeline_stages=3
    )


@pytest.fixture
def resource_limits():
    """Resource limit test cases."""
    return {
        'under_limit': ResourceUtilization(
            luts_used=1000,
            bram_used=100000,
            dsp_used=10
        ),
        'at_limit': ResourceUtilization(
            luts_used=20800,  # Exactly at Artix-7 35T limit
            bram_used=1800000,
            dsp_used=90
        ),
        'over_limit': ResourceUtilization(
            luts_used=25000,  # Over Artix-7 35T limit
            bram_used=2000000,
            dsp_used=100
        )
    }