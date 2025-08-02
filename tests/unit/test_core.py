"""Unit tests for core functionality."""

import pytest
from spiking_fpga.core import FPGATarget


class TestFPGATarget:
    """Test FPGATarget enumeration and methods."""
    
    def test_fpga_target_values(self):
        """Test that all expected FPGA targets are available."""
        expected_targets = {
            "artix7_35t", "artix7_100t", 
            "cyclone5_gx", "cyclone5_gt"
        }
        actual_targets = {target.value for target in FPGATarget}
        assert actual_targets == expected_targets
    
    def test_vendor_property(self):
        """Test vendor property returns correct values."""
        assert FPGATarget.ARTIX7_35T.vendor == "xilinx"
        assert FPGATarget.ARTIX7_100T.vendor == "xilinx"
        assert FPGATarget.CYCLONE_V_GX.vendor == "intel"
        assert FPGATarget.CYCLONE_V_GT.vendor == "intel"
    
    def test_toolchain_property(self):
        """Test toolchain property returns correct values."""
        assert FPGATarget.ARTIX7_35T.toolchain == "vivado"
        assert FPGATarget.ARTIX7_100T.toolchain == "vivado"
        assert FPGATarget.CYCLONE_V_GX.toolchain == "quartus"
        assert FPGATarget.CYCLONE_V_GT.toolchain == "quartus"
    
    def test_resources_property(self):
        """Test resources property returns expected keys."""
        for target in FPGATarget:
            resources = target.resources
            assert "max_neurons" in resources
            assert resources["max_neurons"] > 0
            
            if target.vendor == "xilinx":
                assert "logic_cells" in resources
                assert "bram_kb" in resources
                assert "dsp_slices" in resources
            elif target.vendor == "intel":
                assert "logic_elements" in resources
                assert "m10k_blocks" in resources
                assert "dsp_blocks" in resources
    
    def test_artix7_35t_resources(self):
        """Test specific resource values for Artix-7 35T."""
        resources = FPGATarget.ARTIX7_35T.resources
        assert resources["logic_cells"] == 33280
        assert resources["bram_kb"] == 1800
        assert resources["dsp_slices"] == 90
        assert resources["max_neurons"] == 25000