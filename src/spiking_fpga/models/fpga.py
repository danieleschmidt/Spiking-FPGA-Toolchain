"""
FPGA platform models and resource utilization tracking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List
import math


class FPGAVendor(str, Enum):
    XILINX = "xilinx"
    INTEL = "intel"
    LATTICE = "lattice"
    MICROSEMI = "microsemi"


class FPGAFamily(str, Enum):
    ARTIX7 = "artix7"
    KINTEX7 = "kintex7"
    VIRTEX7 = "virtex7"
    CYCLONE_V = "cyclone_v"
    STRATIX_V = "stratix_v"


@dataclass
class FPGAResources:
    """FPGA resource specifications."""
    logic_cells: int
    luts: int
    flip_flops: int
    bram_blocks: int
    bram_bits: int
    dsp_slices: int
    io_pins: int
    pll_count: int
    max_frequency_mhz: int


class FPGATarget:
    """FPGA target platform specifications."""
    
    # Predefined FPGA targets
    ARTIX7_35T = "artix7_35t"
    ARTIX7_100T = "artix7_100t"
    CYCLONE_V_GX = "cyclone_v_gx"
    CYCLONE_V_GT = "cyclone_v_gt"
    
    FPGA_SPECS = {
        ARTIX7_35T: FPGAResources(
            logic_cells=33280,
            luts=20800,
            flip_flops=41600,
            bram_blocks=50,
            bram_bits=1800000,  # 1.8 Mb
            dsp_slices=90,
            io_pins=250,
            pll_count=5,
            max_frequency_mhz=450
        ),
        ARTIX7_100T: FPGAResources(
            logic_cells=101440,
            luts=63400,
            flip_flops=126800,
            bram_blocks=135,
            bram_bits=4860000,  # 4.9 Mb
            dsp_slices=240,
            io_pins=300,
            pll_count=6,
            max_frequency_mhz=450
        ),
        CYCLONE_V_GX: FPGAResources(
            logic_cells=77000,
            luts=77000,
            flip_flops=77000,
            bram_blocks=200,
            bram_bits=2000000,  # ~2 Mb
            dsp_slices=150,
            io_pins=400,
            pll_count=8,
            max_frequency_mhz=400
        ),
        CYCLONE_V_GT: FPGAResources(
            logic_cells=150000,
            luts=150000,
            flip_flops=150000,
            bram_blocks=400,
            bram_bits=4000000,  # ~4 Mb
            dsp_slices=300,
            io_pins=500,
            pll_count=12,
            max_frequency_mhz=400
        )
    }
    
    def __init__(self, target: str):
        if target not in self.FPGA_SPECS:
            raise ValueError(f"Unknown FPGA target: {target}")
        self.target = target
        self.specs = self.FPGA_SPECS[target]
        self.vendor = FPGAVendor.XILINX if "artix" in target else FPGAVendor.INTEL
        
    def get_max_neurons(self, neuron_type: str = "LIF") -> int:
        """Estimate maximum number of neurons for this FPGA."""
        if neuron_type == "LIF":
            # Conservative estimate: 4 LUTs per LIF neuron
            lut_limited = self.specs.luts // 4
            # Memory limited by neuron state storage (64 bits per neuron)
            mem_limited = self.specs.bram_bits // 64
            return min(lut_limited, mem_limited)
        elif neuron_type == "AdaptiveLIF":
            # More complex: 8 LUTs per neuron
            lut_limited = self.specs.luts // 8
            mem_limited = self.specs.bram_bits // 128
            return min(lut_limited, mem_limited)
        else:
            # Conservative fallback
            return self.specs.luts // 10
    
    def get_max_synapses(self, bits_per_synapse: int = 16) -> int:
        """Estimate maximum number of synapses based on memory."""
        return self.specs.bram_bits // bits_per_synapse
    
    def estimate_power_consumption(self, utilization: 'ResourceUtilization') -> float:
        """Estimate power consumption in watts."""
        # Rough power estimation based on utilization
        base_power = 0.5  # Static power (W)
        
        # Dynamic power scaling
        lut_power = (utilization.luts_used / self.specs.luts) * 0.8
        bram_power = (utilization.bram_used / self.specs.bram_bits) * 0.3
        dsp_power = (utilization.dsp_used / self.specs.dsp_slices) * 0.2
        
        return base_power + lut_power + bram_power + dsp_power


@dataclass 
class ResourceUtilization:
    """FPGA resource utilization tracking."""
    luts_used: int = 0
    flip_flops_used: int = 0
    bram_used: int = 0  # In bits
    dsp_used: int = 0
    io_used: int = 0
    
    def add_utilization(self, other: 'ResourceUtilization'):
        """Add another utilization to this one."""
        self.luts_used += other.luts_used
        self.flip_flops_used += other.flip_flops_used
        self.bram_used += other.bram_used
        self.dsp_used += other.dsp_used
        self.io_used += other.io_used
    
    def get_utilization_percentages(self, target: FPGATarget) -> Dict[str, float]:
        """Get utilization as percentages of target FPGA capacity."""
        specs = target.specs
        return {
            'luts': (self.luts_used / specs.luts) * 100,
            'flip_flops': (self.flip_flops_used / specs.flip_flops) * 100,
            'bram': (self.bram_used / specs.bram_bits) * 100,
            'dsp': (self.dsp_used / specs.dsp_slices) * 100,
            'io': (self.io_used / specs.io_pins) * 100
        }
    
    def check_constraints(self, target: FPGATarget) -> List[str]:
        """Check if utilization exceeds FPGA capacity."""
        violations = []
        specs = target.specs
        
        if self.luts_used > specs.luts:
            violations.append(f"LUT utilization ({self.luts_used}) exceeds capacity ({specs.luts})")
        if self.flip_flops_used > specs.flip_flops:
            violations.append(f"FF utilization ({self.flip_flops_used}) exceeds capacity ({specs.flip_flops})")
        if self.bram_used > specs.bram_bits:
            violations.append(f"BRAM utilization ({self.bram_used}) exceeds capacity ({specs.bram_bits})")
        if self.dsp_used > specs.dsp_slices:
            violations.append(f"DSP utilization ({self.dsp_used}) exceeds capacity ({specs.dsp_slices})")
        if self.io_used > specs.io_pins:
            violations.append(f"IO utilization ({self.io_used}) exceeds capacity ({specs.io_pins})")
            
        return violations


@dataclass
class TimingConstraints:
    """Timing constraints for FPGA implementation."""
    clock_frequency_mhz: float = 100.0
    setup_time_ns: float = 2.0
    hold_time_ns: float = 1.0
    clock_to_output_ns: float = 5.0
    
    # Pipeline constraints
    max_combinational_delay_ns: float = 8.0
    pipeline_stages: int = 3
    
    def validate_constraints(self, target: FPGATarget) -> bool:
        """Validate timing constraints against FPGA capabilities."""
        if self.clock_frequency_mhz > target.specs.max_frequency_mhz:
            raise ValueError(
                f"Clock frequency {self.clock_frequency_mhz}MHz exceeds "
                f"FPGA maximum {target.specs.max_frequency_mhz}MHz"
            )
        
        clock_period_ns = 1000.0 / self.clock_frequency_mhz
        min_period = self.setup_time_ns + self.max_combinational_delay_ns + self.hold_time_ns
        
        if clock_period_ns < min_period:
            raise ValueError(
                f"Clock period {clock_period_ns}ns is too short for "
                f"timing requirements (minimum {min_period}ns)"
            )
        
        return True
    
    def calculate_throughput(self, neurons_per_stage: int) -> float:
        """Calculate spike processing throughput in spikes/second."""
        cycles_per_update = self.pipeline_stages
        updates_per_second = (self.clock_frequency_mhz * 1e6) / cycles_per_update
        return neurons_per_stage * updates_per_second


class ResourceEstimator:
    """Estimates FPGA resource usage for SNN implementations."""
    
    # Resource usage per component (empirical estimates)
    LIF_NEURON_RESOURCES = ResourceUtilization(
        luts_used=4, flip_flops_used=8, bram_used=64, dsp_used=0, io_used=0
    )
    
    ADAPTIVE_LIF_RESOURCES = ResourceUtilization(
        luts_used=8, flip_flops_used=16, bram_used=128, dsp_used=0, io_used=0
    )
    
    STDP_SYNAPSE_RESOURCES = ResourceUtilization(
        luts_used=2, flip_flops_used=4, bram_used=32, dsp_used=0, io_used=0
    )
    
    SPIKE_ROUTER_BASE = ResourceUtilization(
        luts_used=100, flip_flops_used=200, bram_used=1024, dsp_used=0, io_used=0
    )
    
    @classmethod
    def estimate_neuron_resources(cls, neuron_count: int, neuron_type: str) -> ResourceUtilization:
        """Estimate resources for a group of neurons."""
        if neuron_type == "LIF":
            base = cls.LIF_NEURON_RESOURCES
        elif neuron_type == "AdaptiveLIF":
            base = cls.ADAPTIVE_LIF_RESOURCES
        else:
            # Conservative fallback
            base = ResourceUtilization(luts_used=10, flip_flops_used=20, bram_used=128)
        
        return ResourceUtilization(
            luts_used=base.luts_used * neuron_count,
            flip_flops_used=base.flip_flops_used * neuron_count,
            bram_used=base.bram_used * neuron_count,
            dsp_used=base.dsp_used * neuron_count,
            io_used=base.io_used * neuron_count
        )
    
    @classmethod
    def estimate_synapse_resources(cls, synapse_count: int, plasticity_enabled: bool = False) -> ResourceUtilization:
        """Estimate resources for synaptic connections."""
        # Base synapse storage (16 bits per synapse)
        bram_per_synapse = 16
        
        if plasticity_enabled:
            # STDP requires additional state and logic
            base = cls.STDP_SYNAPSE_RESOURCES
            bram_per_synapse += 16  # Additional STDP state storage
        else:
            base = ResourceUtilization(luts_used=0, flip_flops_used=0, bram_used=16)
        
        return ResourceUtilization(
            luts_used=base.luts_used * synapse_count if plasticity_enabled else 0,
            flip_flops_used=base.flip_flops_used * synapse_count if plasticity_enabled else 0,
            bram_used=bram_per_synapse * synapse_count,
            dsp_used=0,
            io_used=0
        )
    
    @classmethod
    def estimate_routing_resources(cls, neuron_count: int, avg_fan_out: int) -> ResourceUtilization:
        """Estimate resources for spike routing infrastructure."""
        # Address-event routing scales logarithmically with network size
        address_bits = math.ceil(math.log2(max(neuron_count, 2)))
        routing_complexity = address_bits * avg_fan_out
        
        base = cls.SPIKE_ROUTER_BASE
        scaling_factor = max(1, routing_complexity // 100)
        
        return ResourceUtilization(
            luts_used=base.luts_used * scaling_factor,
            flip_flops_used=base.flip_flops_used * scaling_factor,
            bram_used=base.bram_used * scaling_factor,
            dsp_used=0,
            io_used=8  # AER interface I/O
        )
    
    @classmethod
    def estimate_total_resources(cls, network_spec: Dict) -> ResourceUtilization:
        """Estimate total resources for complete network implementation."""
        total = ResourceUtilization()
        
        # Estimate neuron resources
        for layer in network_spec.get('layers', []):
            layer_resources = cls.estimate_neuron_resources(
                layer['size'], layer.get('neuron_model', 'LIF')
            )
            total.add_utilization(layer_resources)
        
        # Estimate synapse resources
        total_synapses = 0
        plasticity_enabled = False
        for conn in network_spec.get('connections', []):
            # Estimate number of synapses based on sparsity
            source_size = next(l['size'] for l in network_spec['layers'] 
                             if l['id'] == conn['source_layer'])
            target_size = next(l['size'] for l in network_spec['layers']
                             if l['id'] == conn['target_layer'])
            
            if conn['connectivity'] == 'full':
                synapses = source_size * target_size
            else:
                sparsity = conn.get('sparsity', 0.1)
                synapses = int(source_size * target_size * sparsity)
            
            total_synapses += synapses
            if conn.get('plasticity_enabled', False):
                plasticity_enabled = True
        
        synapse_resources = cls.estimate_synapse_resources(total_synapses, plasticity_enabled)
        total.add_utilization(synapse_resources)
        
        # Estimate routing resources
        total_neurons = sum(layer['size'] for layer in network_spec.get('layers', []))
        avg_fan_out = total_synapses / max(total_neurons, 1)
        routing_resources = cls.estimate_routing_resources(total_neurons, int(avg_fan_out))
        total.add_utilization(routing_resources)
        
        return total