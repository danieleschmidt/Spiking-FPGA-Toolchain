"""
Neuron models and synaptic connection implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class NeuronType(str, Enum):
    LIF = "LIF"
    ADAPTIVE_LIF = "AdaptiveLIF"
    IZHIKEVICH = "Izhikevich"
    HODGKIN_HUXLEY = "HodgkinHuxley"


@dataclass
class NeuronState:
    """Runtime state of a single neuron."""
    membrane_potential: float = 0.0
    adaptation_current: float = 0.0
    refractory_counter: int = 0
    last_spike_time: float = -np.inf
    synaptic_current: float = 0.0


class NeuronModel(ABC):
    """Abstract base class for neuron models."""
    
    def __init__(self, neuron_id: int, parameters: Dict[str, float]):
        self.neuron_id = neuron_id
        self.parameters = parameters
        self.state = NeuronState()
    
    @abstractmethod
    def update(self, dt: float, input_current: float) -> bool:
        """Update neuron state and return True if spike occurred."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset neuron to resting state."""
        pass
    
    @abstractmethod
    def to_hdl_parameters(self) -> Dict[str, int]:
        """Convert floating-point parameters to fixed-point for HDL."""
        pass


class LIFNeuron(NeuronModel):
    """Leaky Integrate-and-Fire neuron model."""
    
    def __init__(self, neuron_id: int, tau_m: float = 20.0, v_thresh: float = 1.0, 
                 v_rest: float = 0.0, v_reset: float = 0.0, refractory_period: float = 2.0):
        parameters = {
            'tau_m': tau_m,
            'v_thresh': v_thresh, 
            'v_rest': v_rest,
            'v_reset': v_reset,
            'refractory_period': refractory_period
        }
        super().__init__(neuron_id, parameters)
    
    def update(self, dt: float, input_current: float) -> bool:
        """Update LIF neuron dynamics."""
        # Check refractory period
        if self.state.refractory_counter > 0:
            self.state.refractory_counter -= 1
            return False
        
        # Update membrane potential
        tau_m = self.parameters['tau_m']
        v_rest = self.parameters['v_rest']
        
        # Leaky integration: dV/dt = (v_rest - V + I)/tau_m
        dv_dt = (v_rest - self.state.membrane_potential + input_current) / tau_m
        self.state.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.state.membrane_potential >= self.parameters['v_thresh']:
            self.state.membrane_potential = self.parameters['v_reset']
            self.state.refractory_counter = int(self.parameters['refractory_period'] / dt)
            return True
        
        return False
    
    def reset(self):
        """Reset neuron to resting state."""
        self.state = NeuronState()
        self.state.membrane_potential = self.parameters['v_rest']
    
    def to_hdl_parameters(self) -> Dict[str, int]:
        """Convert to 16-bit fixed-point parameters for HDL implementation."""
        # Use Q8.8 fixed-point format (8 integer bits, 8 fractional bits)
        scale_factor = 256
        
        return {
            'TAU_M': int(self.parameters['tau_m'] * scale_factor),
            'V_THRESH': int(self.parameters['v_thresh'] * scale_factor),
            'V_REST': int(self.parameters['v_rest'] * scale_factor),
            'V_RESET': int(self.parameters['v_reset'] * scale_factor),
            'REFRAC_PERIOD': int(self.parameters['refractory_period'] * scale_factor)
        }
    
    def generate_verilog_module(self) -> str:
        """Generate Verilog module for this LIF neuron."""
        params = self.to_hdl_parameters()
        
        return f"""
module lif_neuron #(
    parameter TAU_M = {params['TAU_M']},
    parameter V_THRESH = {params['V_THRESH']},
    parameter V_REST = {params['V_REST']},
    parameter V_RESET = {params['V_RESET']},
    parameter REFRAC_PERIOD = {params['REFRAC_PERIOD']}
)(
    input wire clk,
    input wire rst,
    input wire [15:0] input_current,
    input wire update_enable,
    output reg spike_out,
    output wire [15:0] membrane_potential
);

    reg [15:0] v_mem;
    reg [7:0] refrac_counter;
    wire [15:0] leak_current;
    wire [31:0] dv_dt_temp;
    wire [15:0] dv_dt;
    
    // Membrane potential output
    assign membrane_potential = v_mem;
    
    // Calculate leak current: (V_REST - v_mem) / TAU_M
    assign leak_current = (V_REST > v_mem) ? 
                         ((V_REST - v_mem) << 8) / TAU_M :
                         -((v_mem - V_REST) << 8) / TAU_M;
    
    // Calculate membrane potential change
    assign dv_dt_temp = leak_current + input_current;
    assign dv_dt = dv_dt_temp[23:8]; // Scale down from Q16.16 to Q8.8
    
    always @(posedge clk) begin
        if (rst) begin
            v_mem <= V_REST;
            refrac_counter <= 0;
            spike_out <= 0;
        end
        else if (update_enable) begin
            spike_out <= 0;
            
            if (refrac_counter > 0) begin
                // In refractory period
                refrac_counter <= refrac_counter - 1;
            end
            else begin
                // Update membrane potential
                v_mem <= v_mem + dv_dt;
                
                // Check for spike
                if (v_mem >= V_THRESH) begin
                    v_mem <= V_RESET;
                    refrac_counter <= REFRAC_PERIOD[7:0];
                    spike_out <= 1;
                end
            end
        end
    end

endmodule"""


class AdaptiveLIFNeuron(LIFNeuron):
    """Adaptive LIF neuron with spike-rate adaptation."""
    
    def __init__(self, neuron_id: int, tau_m: float = 20.0, tau_adapt: float = 100.0,
                 adaptation_strength: float = 0.1, **kwargs):
        super().__init__(neuron_id, tau_m, **kwargs)
        self.parameters.update({
            'tau_adapt': tau_adapt,
            'adaptation_strength': adaptation_strength
        })
    
    def update(self, dt: float, input_current: float) -> bool:
        """Update adaptive LIF neuron with adaptation current."""
        # Update adaptation current
        tau_adapt = self.parameters['tau_adapt']
        self.state.adaptation_current *= np.exp(-dt / tau_adapt)
        
        # Subtract adaptation current from input
        effective_current = input_current - self.state.adaptation_current
        
        # Standard LIF update
        spike_occurred = super().update(dt, effective_current)
        
        # Increase adaptation current on spike
        if spike_occurred:
            self.state.adaptation_current += self.parameters['adaptation_strength']
        
        return spike_occurred


@dataclass
class SynapticConnection:
    """Represents a synaptic connection between neurons."""
    source_neuron: int
    target_neuron: int
    weight: float
    delay: float = 1.0  # Synaptic delay in milliseconds
    
    # STDP parameters
    stdp_enabled: bool = False
    a_plus: float = 0.1
    a_minus: float = 0.12
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    
    def __post_init__(self):
        """Initialize STDP traces."""
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
    
    def update_weight_stdp(self, dt: float, pre_spike: bool, post_spike: bool, 
                          current_time: float) -> float:
        """Update synaptic weight using STDP rule."""
        if not self.stdp_enabled:
            return self.weight
        
        # Update pre-synaptic trace
        if pre_spike:
            self.pre_trace += self.a_plus
            self.last_pre_spike = current_time
            
            # Check for post-before-pre (depression)
            if current_time - self.last_post_spike < 50.0:  # 50ms STDP window
                delta_t = current_time - self.last_post_spike
                self.weight -= self.a_minus * np.exp(-delta_t / self.tau_minus)
        
        # Update post-synaptic trace  
        if post_spike:
            self.post_trace += self.a_minus
            self.last_post_spike = current_time
            
            # Check for pre-before-post (potentiation)
            if current_time - self.last_pre_spike < 50.0:  # 50ms STDP window
                delta_t = current_time - self.last_pre_spike
                self.weight += self.a_plus * np.exp(-delta_t / self.tau_plus)
        
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)
        
        # Clip weights to reasonable range
        self.weight = np.clip(self.weight, -2.0, 2.0)
        
        return self.weight
    
    def to_hdl_parameters(self) -> Dict[str, int]:
        """Convert connection parameters to fixed-point for HDL."""
        scale_factor = 256
        
        return {
            'WEIGHT': int(self.weight * scale_factor),
            'DELAY': int(self.delay),
            'A_PLUS': int(self.a_plus * scale_factor) if self.stdp_enabled else 0,
            'A_MINUS': int(self.a_minus * scale_factor) if self.stdp_enabled else 0,
            'TAU_PLUS': int(self.tau_plus * scale_factor) if self.stdp_enabled else 0,
            'TAU_MINUS': int(self.tau_minus * scale_factor) if self.stdp_enabled else 0
        }


class SynapseManager:
    """Manages synaptic connections and spike routing."""
    
    def __init__(self):
        self.connections: List[SynapticConnection] = []
        self.spike_buffer: Dict[int, List[Tuple[float, float]]] = {}  # neuron_id -> [(time, weight)]
    
    def add_connection(self, connection: SynapticConnection):
        """Add a synaptic connection."""
        self.connections.append(connection)
        if connection.target_neuron not in self.spike_buffer:
            self.spike_buffer[connection.target_neuron] = []
    
    def propagate_spike(self, source_neuron: int, spike_time: float):
        """Propagate spike through synaptic connections."""
        for conn in self.connections:
            if conn.source_neuron == source_neuron:
                # Add delayed spike to target neuron's buffer
                arrival_time = spike_time + conn.delay
                self.spike_buffer[conn.target_neuron].append((arrival_time, conn.weight))
    
    def get_synaptic_input(self, target_neuron: int, current_time: float, 
                          time_window: float = 0.1) -> float:
        """Get total synaptic input for a neuron at current time."""
        if target_neuron not in self.spike_buffer:
            return 0.0
        
        total_input = 0.0
        remaining_spikes = []
        
        for spike_time, weight in self.spike_buffer[target_neuron]:
            if abs(spike_time - current_time) < time_window:
                total_input += weight
            elif spike_time > current_time:
                remaining_spikes.append((spike_time, weight))
        
        # Keep only future spikes
        self.spike_buffer[target_neuron] = remaining_spikes
        
        return total_input
    
    def generate_routing_table(self) -> Dict[str, List[Dict]]:
        """Generate address-event routing table for FPGA implementation."""
        routing_table = {}
        
        for i, conn in enumerate(self.connections):
            entry = {
                'connection_id': i,
                'source_addr': conn.source_neuron,
                'target_addr': conn.target_neuron,
                'weight': conn.to_hdl_parameters()['WEIGHT'],
                'delay': conn.delay
            }
            
            if conn.source_neuron not in routing_table:
                routing_table[conn.source_neuron] = []
            routing_table[conn.source_neuron].append(entry)
        
        return routing_table