"""Neuron model definitions with HDL generation capabilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math


class NeuronModel(ABC):
    """Abstract base class for neuron models."""
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get neuron parameters."""
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate neuron parameters."""
        pass
    
    @abstractmethod
    def to_hdl(self, neuron_id: int) -> str:
        """Generate HDL code for this neuron model."""
        pass
    
    @abstractmethod
    def get_resource_estimate(self) -> Dict[str, int]:
        """Estimate FPGA resource usage."""
        pass


@dataclass
class LIFNeuron(NeuronModel):
    """Leaky Integrate-and-Fire neuron model."""
    
    v_thresh: float = 1.0      # Threshold voltage
    v_reset: float = 0.0       # Reset voltage 
    tau_m: float = 20.0        # Membrane time constant (ms)
    v_rest: float = 0.0        # Resting potential
    refractory_period: float = 1.0  # Refractory period (ms)
    
    def get_parameters(self) -> Dict[str, float]:
        """Get neuron parameters."""
        return {
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "tau_m": self.tau_m,
            "v_rest": self.v_rest,
            "refractory_period": self.refractory_period,
        }
    
    def validate_parameters(self) -> bool:
        """Validate neuron parameters."""
        if self.v_thresh <= self.v_reset:
            raise ValueError("Threshold must be greater than reset voltage")
        if self.tau_m <= 0:
            raise ValueError("Time constant must be positive")
        if self.refractory_period < 0:
            raise ValueError("Refractory period must be non-negative")
        return True
    
    def to_hdl(self, neuron_id: int) -> str:
        """Generate Verilog HDL for LIF neuron."""
        # Convert floating point parameters to fixed point (16.8 format)
        thresh_fp = int(self.v_thresh * 256)
        reset_fp = int(self.v_reset * 256) 
        rest_fp = int(self.v_rest * 256)
        decay_fp = int((1.0 - 1.0/self.tau_m) * 256)
        refrac_cycles = int(self.refractory_period)
        
        hdl_code = f\"\"\"
// LIF Neuron #{neuron_id}
module lif_neuron_{neuron_id} #(
    parameter V_THRESH = {thresh_fp},
    parameter V_RESET = {reset_fp},
    parameter V_REST = {rest_fp}, 
    parameter DECAY_FACTOR = {decay_fp},
    parameter REFRAC_CYCLES = {refrac_cycles}
) (
    input wire clk,
    input wire rst,
    input wire [15:0] current_in,
    input wire spike_in_valid,
    output reg spike_out,
    output reg [15:0] voltage_out
);

    // Internal state
    reg signed [23:0] membrane_voltage;
    reg [7:0] refractory_counter;
    reg in_refractory;
    
    always @(posedge clk) begin
        if (rst) begin
            membrane_voltage <= V_REST << 8;
            spike_out <= 1'b0;
            refractory_counter <= 8'b0;
            in_refractory <= 1'b0;
        end else begin
            spike_out <= 1'b0;
            
            // Handle refractory period
            if (in_refractory) begin
                if (refractory_counter > 0) begin
                    refractory_counter <= refractory_counter - 1;
                end else begin
                    in_refractory <= 1'b0;
                    membrane_voltage <= V_REST << 8;
                end
            end else begin
                // Membrane voltage decay
                membrane_voltage <= (membrane_voltage * DECAY_FACTOR) >> 8;
                
                // Add input current
                if (spike_in_valid) begin
                    membrane_voltage <= membrane_voltage + (current_in << 8);
                end
                
                // Check for spike threshold
                if (membrane_voltage >= (V_THRESH << 8)) begin
                    spike_out <= 1'b1;
                    membrane_voltage <= V_RESET << 8;
                    in_refractory <= 1'b1;
                    refractory_counter <= REFRAC_CYCLES;
                end
            end
        end
    end
    
    // Output current voltage (converted back to 16.8 format)
    assign voltage_out = membrane_voltage[23:8];

endmodule
\"\"\"
        return hdl_code.strip()
    
    def get_resource_estimate(self) -> Dict[str, int]:
        """Estimate FPGA resource usage for LIF neuron."""
        return {
            "luts": 45,        # Logic LUTs for arithmetic and control
            "registers": 35,    # Flip-flops for state storage
            "bram_bits": 0,    # No block RAM needed
            "dsp_slices": 1,   # One DSP for multiplication
        }


@dataclass 
class IzhikevichNeuron(NeuronModel):
    """Izhikevich neuron model with rich dynamics."""
    
    a: float = 0.02    # Recovery time constant
    b: float = 0.2     # Sensitivity to sub-threshold fluctuations
    c: float = -65.0   # After-spike reset value
    d: float = 8.0     # After-spike reset of recovery variable
    
    def get_parameters(self) -> Dict[str, float]:
        """Get neuron parameters."""
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}
    
    def validate_parameters(self) -> bool:
        """Validate neuron parameters."""
        if self.a <= 0 or self.b <= 0:
            raise ValueError("Parameters a and b must be positive")
        return True
    
    def to_hdl(self, neuron_id: int) -> str:
        """Generate Verilog HDL for Izhikevich neuron."""
        # Convert to fixed point (8.8 format for better precision)
        a_fp = int(self.a * 256)
        b_fp = int(self.b * 256)
        c_fp = int(self.c * 256)
        d_fp = int(self.d * 256)
        
        hdl_code = f\"\"\"
// Izhikevich Neuron #{neuron_id}
module izhikevich_neuron_{neuron_id} #(
    parameter A = {a_fp},
    parameter B = {b_fp}, 
    parameter C = {c_fp},
    parameter D = {d_fp}
) (
    input wire clk,
    input wire rst,
    input wire [15:0] current_in,
    input wire spike_in_valid,
    output reg spike_out,
    output reg [15:0] voltage_out
);

    // Internal state variables
    reg signed [23:0] v;  // Membrane potential
    reg signed [23:0] u;  // Recovery variable
    
    // Intermediate calculations
    wire signed [47:0] v_squared;
    wire signed [31:0] v_term, u_term;
    
    assign v_squared = v * v;
    assign v_term = (v_squared[39:16] + (5 * v) + (140 << 8));
    assign u_term = (u * A) >> 8;
    
    always @(posedge clk) begin
        if (rst) begin
            v <= -65 << 8;  // Resting potential
            u <= 0;
            spike_out <= 1'b0;
        end else begin
            spike_out <= 1'b0;
            
            // Check for spike
            if (v >= (30 << 8)) begin  // Spike threshold at 30mV
                spike_out <= 1'b1;
                v <= C;
                u <= u + D;
            end else begin
                // Update membrane potential: dv/dt = 0.04v² + 5v + 140 - u + I
                v <= v + ((v_term - u + (current_in << 8)) >> 6);  // Timestep scaling
                
                // Update recovery variable: du/dt = a(bv - u)
                u <= u + (((B * v) >> 8) - u_term);
            end
        end
    end
    
    assign voltage_out = v[23:8];

endmodule
\"\"\"
        return hdl_code.strip()
    
    def get_resource_estimate(self) -> Dict[str, int]:
        """Estimate FPGA resource usage for Izhikevich neuron."""
        return {
            "luts": 120,       # More complex arithmetic
            "registers": 50,    # Additional state variables
            "bram_bits": 0,
            "dsp_slices": 3,   # Multiple multiplications needed
        }


@dataclass
class AdaptiveLIFNeuron(NeuronModel):
    """Adaptive LIF neuron with spike frequency adaptation."""
    
    v_thresh: float = 1.0
    v_reset: float = 0.0
    tau_m: float = 20.0
    tau_adapt: float = 100.0   # Adaptation time constant
    delta_thresh: float = 0.1   # Threshold increase per spike
    
    def get_parameters(self) -> Dict[str, float]:
        """Get neuron parameters."""
        return {
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "tau_m": self.tau_m,
            "tau_adapt": self.tau_adapt,
            "delta_thresh": self.delta_thresh,
        }
    
    def validate_parameters(self) -> bool:
        """Validate neuron parameters."""
        if self.tau_m <= 0 or self.tau_adapt <= 0:
            raise ValueError("Time constants must be positive")
        if self.delta_thresh < 0:
            raise ValueError("Threshold adaptation must be non-negative")
        return True
    
    def to_hdl(self, neuron_id: int) -> str:
        """Generate Verilog HDL for adaptive LIF neuron."""
        thresh_fp = int(self.v_thresh * 256)
        reset_fp = int(self.v_reset * 256)
        decay_fp = int((1.0 - 1.0/self.tau_m) * 256)
        adapt_decay_fp = int((1.0 - 1.0/self.tau_adapt) * 256)
        delta_thresh_fp = int(self.delta_thresh * 256)
        
        hdl_code = f\"\"\"
// Adaptive LIF Neuron #{neuron_id}
module adaptive_lif_neuron_{neuron_id} #(
    parameter V_THRESH_BASE = {thresh_fp},
    parameter V_RESET = {reset_fp},
    parameter DECAY_FACTOR = {decay_fp},
    parameter ADAPT_DECAY = {adapt_decay_fp},
    parameter DELTA_THRESH = {delta_thresh_fp}
) (
    input wire clk,
    input wire rst,
    input wire [15:0] current_in,
    input wire spike_in_valid,
    output reg spike_out,
    output reg [15:0] voltage_out,
    output reg [15:0] threshold_out
);

    // Internal state
    reg signed [23:0] membrane_voltage;
    reg [23:0] adaptive_threshold;
    reg [23:0] current_threshold;
    
    always @(posedge clk) begin
        if (rst) begin
            membrane_voltage <= 0;
            adaptive_threshold <= 0;
            current_threshold <= V_THRESH_BASE << 8;
            spike_out <= 1'b0;
        end else begin
            spike_out <= 1'b0;
            
            // Update current threshold (base + adaptive component)
            current_threshold <= (V_THRESH_BASE << 8) + adaptive_threshold;
            
            // Membrane voltage decay
            membrane_voltage <= (membrane_voltage * DECAY_FACTOR) >> 8;
            
            // Adaptive threshold decay
            adaptive_threshold <= (adaptive_threshold * ADAPT_DECAY) >> 8;
            
            // Add input current
            if (spike_in_valid) begin
                membrane_voltage <= membrane_voltage + (current_in << 8);
            end
            
            // Check for spike threshold
            if (membrane_voltage >= current_threshold) begin
                spike_out <= 1'b1;
                membrane_voltage <= V_RESET << 8;
                // Increase adaptive threshold
                adaptive_threshold <= adaptive_threshold + (DELTA_THRESH << 8);
            end
        end
    end
    
    assign voltage_out = membrane_voltage[23:8];
    assign threshold_out = current_threshold[23:8];

endmodule
\"\"\"
        return hdl_code.strip()
    
    def get_resource_estimate(self) -> Dict[str, int]:
        """Estimate FPGA resource usage for adaptive LIF neuron."""
        return {
            "luts": 75,        # Additional logic for adaptation
            "registers": 55,    # More state variables
            "bram_bits": 0,
            "dsp_slices": 2,   # Multiple multiplications
        }