"""Core types and enumerations for the Spiking-FPGA-Toolchain."""

from enum import Enum
from typing import Dict, Any


class FPGATarget(Enum):
    """Supported FPGA target platforms."""
    
    ARTIX7_35T = "artix7_35t"
    ARTIX7_100T = "artix7_100t"
    CYCLONE_V_GX = "cyclone5_gx"
    CYCLONE_V_GT = "cyclone5_gt"
    
    @property
    def vendor(self) -> str:
        """Get the FPGA vendor for this target."""
        if self.value.startswith("artix7"):
            return "xilinx"
        elif self.value.startswith("cyclone"):
            return "intel"
        else:
            raise ValueError(f"Unknown vendor for target: {self}")
    
    @property
    def toolchain(self) -> str:
        """Get the required toolchain for this target."""
        vendor_map = {
            "xilinx": "vivado",
            "intel": "quartus"
        }
        return vendor_map[self.vendor]
    
    @property
    def resources(self) -> Dict[str, Any]:
        """Get resource constraints for this target platform."""
        resource_map = {
            "artix7_35t": {
                "logic_cells": 33280,
                "bram_kb": 1800,
                "dsp_slices": 90,
                "max_neurons": 25000,
            },
            "artix7_100t": {
                "logic_cells": 101440,
                "bram_kb": 4860,
                "dsp_slices": 240,
                "max_neurons": 100000,
            },
            "cyclone5_gx": {
                "logic_elements": 77000,
                "m10k_blocks": 294,
                "dsp_blocks": 150,
                "max_neurons": 80000,
            },
            "cyclone5_gt": {
                "logic_elements": 149500,
                "m10k_blocks": 647,
                "dsp_blocks": 288,
                "max_neurons": 120000,
            },
        }
        return resource_map.get(self.value, {})