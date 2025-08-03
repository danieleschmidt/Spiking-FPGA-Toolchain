"""Unit tests for core functionality."""

import pytest
import numpy as np
from src.spiking_fpga.models.fpga import FPGATarget, ResourceUtilization, ResourceEstimator
from src.spiking_fpga.models.network import SNNNetwork, Layer, Connection, LayerType, ConnectivityPattern
from src.spiking_fpga.models.neuron import LIFNeuron, AdaptiveLIFNeuron, SynapticConnection


class TestFPGATarget:
    """Test FPGA target functionality."""
    
    def test_fpga_target_creation(self):
        """Test FPGA target instantiation."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        assert target.target == FPGATarget.ARTIX7_35T
        assert target.vendor.value == "xilinx"
        assert target.specs.logic_cells == 33280
    
    def test_max_neurons_estimation(self):
        """Test maximum neuron estimation."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        max_lif = target.get_max_neurons("LIF")
        max_adaptive = target.get_max_neurons("AdaptiveLIF")
        
        assert max_lif > 0
        assert max_adaptive > 0
        assert max_adaptive < max_lif  # Adaptive neurons use more resources
    
    def test_max_synapses_estimation(self):
        """Test maximum synapse estimation."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        max_synapses = target.get_max_synapses(16)
        assert max_synapses > 0
        assert max_synapses == target.specs.bram_bits // 16
    
    def test_power_estimation(self):
        """Test power consumption estimation."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        utilization = ResourceUtilization(
            luts_used=10000,
            bram_used=500000,
            dsp_used=45
        )
        
        power = target.estimate_power_consumption(utilization)
        assert power > 0
        assert power < 10.0  # Reasonable power range
    
    def test_all_targets_valid(self):
        """Test all predefined FPGA targets are valid."""
        for target_name in FPGATarget.FPGA_SPECS.keys():
            target = FPGATarget(target_name)
            assert target.specs.logic_cells > 0
            assert target.specs.bram_bits > 0


class TestResourceUtilization:
    """Test resource utilization tracking."""
    
    def test_resource_creation(self):
        """Test resource utilization creation."""
        resources = ResourceUtilization(
            luts_used=1000,
            flip_flops_used=2000,
            bram_used=500000,
            dsp_used=10
        )
        
        assert resources.luts_used == 1000
        assert resources.flip_flops_used == 2000
        assert resources.bram_used == 500000
        assert resources.dsp_used == 10
    
    def test_add_utilization(self):
        """Test adding resource utilizations."""
        res1 = ResourceUtilization(luts_used=1000, bram_used=100000)
        res2 = ResourceUtilization(luts_used=500, bram_used=50000)
        
        res1.add_utilization(res2)
        
        assert res1.luts_used == 1500
        assert res1.bram_used == 150000
    
    def test_utilization_percentages(self):
        """Test utilization percentage calculation."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        resources = ResourceUtilization(
            luts_used=10400,  # 50% of Artix-7 35T
            bram_used=900000,  # 50% of Artix-7 35T
            dsp_used=45       # 50% of Artix-7 35T
        )
        
        percentages = resources.get_utilization_percentages(target)
        
        assert abs(percentages['luts'] - 50.0) < 1.0
        assert abs(percentages['bram'] - 50.0) < 1.0
        assert abs(percentages['dsp'] - 50.0) < 1.0
    
    def test_constraint_violations(self):
        """Test constraint violation detection."""
        target = FPGATarget(FPGATarget.ARTIX7_35T)
        
        # Within limits
        resources_ok = ResourceUtilization(luts_used=10000, bram_used=500000)
        violations_ok = resources_ok.check_constraints(target)
        assert len(violations_ok) == 0
        
        # Over limits
        resources_bad = ResourceUtilization(
            luts_used=50000,    # Over limit
            bram_used=5000000   # Over limit
        )
        violations_bad = resources_bad.check_constraints(target)
        assert len(violations_bad) > 0


class TestResourceEstimator:
    """Test resource estimation algorithms."""
    
    def test_neuron_resource_estimation(self):
        """Test neuron resource estimation."""
        lif_resources = ResourceEstimator.estimate_neuron_resources(100, "LIF")
        adaptive_resources = ResourceEstimator.estimate_neuron_resources(100, "AdaptiveLIF")
        
        assert lif_resources.luts_used > 0
        assert adaptive_resources.luts_used > lif_resources.luts_used
        assert lif_resources.bram_used > 0
    
    def test_synapse_resource_estimation(self):
        """Test synapse resource estimation."""
        resources_simple = ResourceEstimator.estimate_synapse_resources(1000, False)
        resources_stdp = ResourceEstimator.estimate_synapse_resources(1000, True)
        
        assert resources_simple.bram_used > 0
        assert resources_stdp.bram_used > resources_simple.bram_used
        assert resources_stdp.luts_used > 0  # STDP requires logic
    
    def test_routing_resource_estimation(self):
        """Test routing resource estimation."""
        resources = ResourceEstimator.estimate_routing_resources(1000, 10)
        
        assert resources.luts_used > 0
        assert resources.flip_flops_used > 0
        assert resources.bram_used > 0


class TestNeuronModels:
    """Test neuron model implementations."""
    
    def test_lif_neuron_creation(self):
        """Test LIF neuron creation."""
        neuron = LIFNeuron(neuron_id=0, tau_m=20.0, v_thresh=1.0)
        
        assert neuron.neuron_id == 0
        assert neuron.parameters['tau_m'] == 20.0
        assert neuron.parameters['v_thresh'] == 1.0
        assert neuron.state.membrane_potential == 0.0
    
    def test_lif_neuron_update(self):
        """Test LIF neuron dynamics."""
        neuron = LIFNeuron(neuron_id=0, tau_m=20.0, v_thresh=1.0)
        
        # No spike with small input
        spike = neuron.update(dt=0.1, input_current=0.1)
        assert not spike
        assert neuron.state.membrane_potential > 0
        
        # Spike with large input
        neuron.reset()
        spike = neuron.update(dt=0.1, input_current=100.0)
        assert spike
        assert neuron.state.refractory_counter > 0
    
    def test_adaptive_lif_neuron(self):
        """Test adaptive LIF neuron."""
        neuron = AdaptiveLIFNeuron(
            neuron_id=0, 
            tau_m=20.0, 
            tau_adapt=100.0,
            adaptation_strength=0.1
        )
        
        # First spike
        spike1 = neuron.update(dt=0.1, input_current=100.0)
        assert spike1
        
        # Adaptation current should be non-zero
        assert neuron.state.adaptation_current > 0
        
        # Reset for next spike
        neuron.reset()
        neuron.state.adaptation_current = 0.1  # Some adaptation
        
        # Should require more current due to adaptation
        spike2 = neuron.update(dt=0.1, input_current=100.0)
        # This may or may not spike depending on adaptation strength
    
    def test_neuron_hdl_parameters(self):
        """Test HDL parameter conversion."""
        neuron = LIFNeuron(neuron_id=0, tau_m=20.0, v_thresh=1.0)
        hdl_params = neuron.to_hdl_parameters()
        
        assert 'TAU_M' in hdl_params
        assert 'V_THRESH' in hdl_params
        assert hdl_params['TAU_M'] == int(20.0 * 256)  # Q8.8 format
        assert hdl_params['V_THRESH'] == int(1.0 * 256)
    
    def test_verilog_generation(self):
        """Test Verilog module generation."""
        neuron = LIFNeuron(neuron_id=0)
        verilog = neuron.generate_verilog_module()
        
        assert 'module lif_neuron' in verilog
        assert 'input wire clk' in verilog
        assert 'output reg spike_out' in verilog
        assert 'endmodule' in verilog


class TestSynapticConnection:
    """Test synaptic connection functionality."""
    
    def test_connection_creation(self):
        """Test synaptic connection creation."""
        conn = SynapticConnection(
            source_neuron=0,
            target_neuron=1,
            weight=0.5,
            delay=1.0
        )
        
        assert conn.source_neuron == 0
        assert conn.target_neuron == 1
        assert conn.weight == 0.5
        assert conn.delay == 1.0
    
    def test_stdp_weight_update(self):
        """Test STDP weight updates."""
        conn = SynapticConnection(
            source_neuron=0,
            target_neuron=1,
            weight=0.5,
            stdp_enabled=True,
            a_plus=0.1,
            a_minus=0.12
        )
        
        original_weight = conn.weight
        
        # Pre-before-post (potentiation)
        conn.update_weight_stdp(dt=0.1, pre_spike=True, post_spike=False, current_time=10.0)
        conn.update_weight_stdp(dt=0.1, pre_spike=False, post_spike=True, current_time=15.0)
        
        # Weight should increase
        assert conn.weight > original_weight
    
    def test_hdl_parameters(self):
        """Test HDL parameter conversion for connections."""
        conn = SynapticConnection(
            source_neuron=0,
            target_neuron=1,
            weight=0.5,
            delay=2.0,
            stdp_enabled=True
        )
        
        params = conn.to_hdl_parameters()
        
        assert 'WEIGHT' in params
        assert 'DELAY' in params
        assert 'A_PLUS' in params
        assert params['WEIGHT'] == int(0.5 * 256)
        assert params['DELAY'] == 2