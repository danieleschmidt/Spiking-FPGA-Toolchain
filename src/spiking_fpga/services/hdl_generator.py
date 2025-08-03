"""
HDL generation service for creating Verilog/VHDL from optimized SNN networks.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import math

from ..models.network import SNNNetwork, Layer, Connection
from ..models.fpga import FPGATarget, TimingConstraints
from ..models.neuron import LIFNeuron
from .resource_mapper import PlacementResult


class VerilogTemplate:
    """Manages Verilog HDL templates for SNN components."""
    
    @staticmethod
    def generate_lif_neuron_module(layer: Layer, neuron_id: int = 0) -> str:
        """Generate Verilog module for LIF neuron."""
        # Convert parameters to fixed-point
        scale_factor = 256  # Q8.8 format
        tau_m_fp = int(layer.tau_m * scale_factor)
        tau_syn_fp = int(layer.tau_syn * scale_factor)
        
        return f"""
module lif_neuron_{layer.id}_n{neuron_id} #(
    parameter TAU_M = {tau_m_fp},
    parameter TAU_SYN = {tau_syn_fp},
    parameter V_THRESH = {int(1.0 * scale_factor)},
    parameter V_REST = 0,
    parameter V_RESET = 0,
    parameter REFRAC_PERIOD = {int(2.0 * scale_factor)}
)(
    input wire clk,
    input wire rst,
    input wire [15:0] synaptic_input,
    input wire update_enable,
    output reg spike_out,
    output wire [15:0] membrane_potential,
    output wire [15:0] synaptic_current
);

    // Internal state registers
    reg [15:0] v_mem;
    reg [15:0] i_syn;
    reg [7:0] refrac_counter;
    
    // Combinational logic for membrane dynamics
    wire [31:0] leak_term;
    wire [31:0] syn_decay;
    wire [15:0] dv_dt;
    wire [15:0] di_syn_dt;
    
    // Membrane potential leak: (V_REST - v_mem) / TAU_M
    assign leak_term = (V_REST > v_mem) ? 
                      ((V_REST - v_mem) << 8) / TAU_M :
                      -((v_mem - V_REST) << 8) / TAU_M;
    
    // Synaptic current decay: -i_syn / TAU_SYN
    assign syn_decay = (i_syn << 8) / TAU_SYN;
    assign di_syn_dt = synaptic_input - syn_decay[23:8];
    
    // Membrane potential change: leak + synaptic current
    assign dv_dt = leak_term[23:8] + i_syn;
    
    // Outputs
    assign membrane_potential = v_mem;
    assign synaptic_current = i_syn;
    
    always @(posedge clk) begin
        if (rst) begin
            v_mem <= V_REST;
            i_syn <= 0;
            refrac_counter <= 0;
            spike_out <= 0;
        end
        else if (update_enable) begin
            spike_out <= 0;
            
            // Update synaptic current
            i_syn <= i_syn + di_syn_dt;
            
            if (refrac_counter > 0) begin
                // In refractory period
                refrac_counter <= refrac_counter - 1;
                v_mem <= V_RESET;
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

    @staticmethod
    def generate_spike_router(network: SNNNetwork, routing_table: Dict) -> str:
        """Generate address-event routing module."""
        # Calculate address width needed
        total_neurons = sum(layer.size for layer in network.layers)
        addr_width = max(8, math.ceil(math.log2(total_neurons)))
        
        return f"""
module spike_router #(
    parameter ADDR_WIDTH = {addr_width},
    parameter MAX_CONNECTIONS = 1024
)(
    input wire clk,
    input wire rst,
    
    // Input spike interface
    input wire spike_valid_in,
    input wire [ADDR_WIDTH-1:0] spike_addr_in,
    input wire [15:0] spike_timestamp,
    output reg spike_ready,
    
    // Output spike interface
    output reg spike_valid_out,
    output reg [ADDR_WIDTH-1:0] spike_addr_out,
    output reg [15:0] spike_weight,
    output reg [7:0] spike_delay,
    input wire spike_ready_out,
    
    // Configuration interface
    input wire config_enable,
    input wire [15:0] config_addr,
    input wire [31:0] config_data
);

    // Routing table memory
    reg [31:0] routing_table [0:MAX_CONNECTIONS-1];
    reg [15:0] connection_count;
    
    // State machine
    reg [2:0] state;
    reg [15:0] lookup_index;
    reg [15:0] connections_found;
    
    localparam IDLE = 3'b000;
    localparam LOOKUP = 3'b001;
    localparam OUTPUT = 3'b010;
    
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            spike_valid_out <= 0;
            spike_ready <= 1;
            lookup_index <= 0;
            connections_found <= 0;
        end
        else begin
            case (state)
                IDLE: begin
                    spike_ready <= 1;
                    spike_valid_out <= 0;
                    
                    if (spike_valid_in && spike_ready) begin
                        // Start routing lookup
                        state <= LOOKUP;
                        lookup_index <= 0;
                        connections_found <= 0;
                        spike_ready <= 0;
                    end
                end
                
                LOOKUP: begin
                    // Search routing table for connections from source
                    if (lookup_index < connection_count) begin
                        if (routing_table[lookup_index][31:16] == spike_addr_in) begin
                            // Found connection, prepare output
                            spike_addr_out <= routing_table[lookup_index][15:8];
                            spike_weight <= routing_table[lookup_index][7:0];
                            spike_delay <= 1; // Fixed delay for now
                            state <= OUTPUT;
                        end
                        else begin
                            lookup_index <= lookup_index + 1;
                        end
                    end
                    else begin
                        // Finished lookup
                        state <= IDLE;
                    end
                end
                
                OUTPUT: begin
                    spike_valid_out <= 1;
                    if (spike_ready_out) begin
                        spike_valid_out <= 0;
                        lookup_index <= lookup_index + 1;
                        state <= LOOKUP;
                    end
                end
            endcase
        end
    end
    
    // Configuration interface
    always @(posedge clk) begin
        if (rst) begin
            connection_count <= 0;
        end
        else if (config_enable) begin
            if (config_addr == 16'hFFFF) begin
                // Set connection count
                connection_count <= config_data[15:0];
            end
            else begin
                // Configure routing entry
                routing_table[config_addr] <= config_data;
            end
        end
    end

endmodule"""

    @staticmethod
    def generate_layer_module(layer: Layer, connections: List[Connection]) -> str:
        """Generate module for a complete layer of neurons."""
        return f"""
module layer_{layer.id} #(
    parameter LAYER_SIZE = {layer.size},
    parameter ADDR_WIDTH = 16
)(
    input wire clk,
    input wire rst,
    input wire update_enable,
    
    // Spike input from router
    input wire spike_in_valid,
    input wire [ADDR_WIDTH-1:0] spike_in_addr,
    input wire [15:0] spike_in_weight,
    
    // Spike output to router
    output wire spike_out_valid,
    output wire [ADDR_WIDTH-1:0] spike_out_addr,
    
    // Debug/monitoring
    output wire [LAYER_SIZE-1:0] spike_debug,
    output wire [15:0] membrane_potential_debug [0:LAYER_SIZE-1]
);

    // Neuron instances
    wire [LAYER_SIZE-1:0] neuron_spikes;
    wire [15:0] neuron_v_mem [0:LAYER_SIZE-1];
    reg [15:0] synaptic_inputs [0:LAYER_SIZE-1];
    
    genvar i;
    generate
        for (i = 0; i < LAYER_SIZE; i = i + 1) begin : neuron_array
            lif_neuron_{layer.id}_n0 neuron_inst (
                .clk(clk),
                .rst(rst),
                .synaptic_input(synaptic_inputs[i]),
                .update_enable(update_enable),
                .spike_out(neuron_spikes[i]),
                .membrane_potential(neuron_v_mem[i]),
                .synaptic_current()
            );
        end
    endgenerate
    
    // Input spike distribution
    integer j;
    always @(posedge clk) begin
        if (rst) begin
            for (j = 0; j < LAYER_SIZE; j = j + 1) begin
                synaptic_inputs[j] <= 0;
            end
        end
        else begin
            // Clear previous inputs
            for (j = 0; j < LAYER_SIZE; j = j + 1) begin
                synaptic_inputs[j] <= 0;
            end
            
            // Distribute incoming spike
            if (spike_in_valid && spike_in_addr < LAYER_SIZE) begin
                synaptic_inputs[spike_in_addr] <= spike_in_weight;
            end
        end
    end
    
    // Output spike encoding
    reg [ADDR_WIDTH-1:0] spike_encoder_addr;
    reg spike_encoder_valid;
    
    always @(posedge clk) begin
        if (rst) begin
            spike_encoder_valid <= 0;
            spike_encoder_addr <= 0;
        end
        else begin
            spike_encoder_valid <= 0;
            
            // Priority encoder for output spikes
            for (j = 0; j < LAYER_SIZE; j = j + 1) begin
                if (neuron_spikes[j]) begin
                    spike_encoder_valid <= 1;
                    spike_encoder_addr <= j;
                end
            end
        end
    end
    
    assign spike_out_valid = spike_encoder_valid;
    assign spike_out_addr = spike_encoder_addr;
    assign spike_debug = neuron_spikes;
    
    // Debug assignments
    generate
        for (i = 0; i < LAYER_SIZE; i = i + 1) begin
            assign membrane_potential_debug[i] = neuron_v_mem[i];
        end
    endgenerate

endmodule"""


class HDLGenerator:
    """Generates HDL implementations from optimized SNN networks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.custom_neuron_models = {}
    
    def register_neuron_model(self, model_name: str, model_class: type):
        """Register a custom neuron model for HDL generation."""
        self.custom_neuron_models[model_name] = model_class
    
    def generate_implementation(self, network: SNNNetwork, target: FPGATarget,
                              placement: PlacementResult, timing: TimingConstraints) -> Dict[str, Any]:
        """
        Generate complete HDL implementation for the network.
        
        Args:
            network: Optimized SNN network
            target: Target FPGA platform
            placement: Resource placement results
            timing: Timing constraints
            
        Returns:
            Dictionary containing generated HDL modules and files
        """
        self.logger.info(f"Generating HDL for network '{network.name}'")
        
        result = {
            'modules': {},
            'top_module': '',
            'xdc_constraints': '',
            'sdc_constraints': '',
            'testbench': '',
            'stimulus_data': []
        }
        
        # Generate neuron modules for each layer
        for layer in network.layers:
            self.logger.debug(f"Generating HDL for layer '{layer.id}'")
            
            # Generate individual neuron module
            neuron_module = VerilogTemplate.generate_lif_neuron_module(layer)
            result['modules'][f'lif_neuron_{layer.id}_n0'] = neuron_module
            
            # Generate layer module
            layer_connections = [conn for conn in network.connections 
                               if conn.target_layer == layer.id]
            layer_module = VerilogTemplate.generate_layer_module(layer, layer_connections)
            result['modules'][f'layer_{layer.id}'] = layer_module
        
        # Generate routing infrastructure
        routing_table = self._build_routing_table(network)
        router_module = VerilogTemplate.generate_spike_router(network, routing_table)
        result['modules']['spike_router'] = router_module
        
        # Generate top-level module
        result['top_module'] = self._generate_top_module(network, target, timing)
        
        # Generate constraint files
        result['xdc_constraints'] = self._generate_xdc_constraints(target, timing)
        result['sdc_constraints'] = self._generate_sdc_constraints(target, timing)
        
        # Generate testbench
        result['testbench'] = self._generate_testbench(network)
        result['stimulus_data'] = self._generate_stimulus_data(network)
        
        self.logger.info(f"Generated {len(result['modules'])} HDL modules")
        return result
    
    def _build_routing_table(self, network: SNNNetwork) -> Dict[str, Any]:
        """Build address-event routing table for the network."""
        routing_table = {}
        
        # Assign address ranges to layers
        layer_addresses = {}
        current_addr = 0
        
        for layer in network.layers:
            layer_addresses[layer.id] = {
                'start': current_addr,
                'end': current_addr + layer.size - 1,
                'size': layer.size
            }
            current_addr += layer.size
        
        # Build connection routing entries
        connection_id = 0
        for conn in network.connections:
            source_layer = next(l for l in network.layers if l.id == conn.source_layer)
            target_layer = next(l for l in network.layers if l.id == conn.target_layer)
            
            source_addr_start = layer_addresses[conn.source_layer]['start']
            target_addr_start = layer_addresses[conn.target_layer]['start']
            
            # Generate weight matrix
            weights = conn.generate_weight_matrix(source_layer.size, target_layer.size)
            
            # Add entries for non-zero connections
            for i in range(source_layer.size):
                for j in range(target_layer.size):
                    if abs(weights[i, j]) > 0.01:  # Skip very small weights
                        entry = {
                            'source_addr': source_addr_start + i,
                            'target_addr': target_addr_start + j,
                            'weight': int(weights[i, j] * 256),  # Fixed-point
                            'delay': 1,
                            'connection_id': connection_id
                        }
                        
                        if entry['source_addr'] not in routing_table:
                            routing_table[entry['source_addr']] = []
                        routing_table[entry['source_addr']].append(entry)
                        connection_id += 1
        
        routing_table['layer_addresses'] = layer_addresses
        routing_table['total_connections'] = connection_id
        
        return routing_table
    
    def _generate_top_module(self, network: SNNNetwork, target: FPGATarget, 
                           timing: TimingConstraints) -> str:
        """Generate top-level module integrating all components."""
        # Calculate total neurons for address width
        total_neurons = sum(layer.size for layer in network.layers)
        addr_width = max(8, math.ceil(math.log2(total_neurons)))
        
        # Generate layer instantiation code
        layer_instances = ""
        layer_wires = ""
        
        for i, layer in enumerate(network.layers):
            layer_instances += f"""
    // Layer {layer.id} instantiation
    layer_{layer.id} layer_{layer.id}_inst (
        .clk(clk),
        .rst(rst),
        .update_enable(update_enable),
        .spike_in_valid(layer_{layer.id}_spike_in_valid),
        .spike_in_addr(layer_{layer.id}_spike_in_addr),
        .spike_in_weight(layer_{layer.id}_spike_in_weight),
        .spike_out_valid(layer_{layer.id}_spike_out_valid),
        .spike_out_addr(layer_{layer.id}_spike_out_addr),
        .spike_debug(layer_{layer.id}_spike_debug),
        .membrane_potential_debug(layer_{layer.id}_v_mem_debug)
    );
"""
            
            layer_wires += f"""
    wire layer_{layer.id}_spike_in_valid;
    wire [{addr_width-1}:0] layer_{layer.id}_spike_in_addr;
    wire [15:0] layer_{layer.id}_spike_in_weight;
    wire layer_{layer.id}_spike_out_valid;
    wire [{addr_width-1}:0] layer_{layer.id}_spike_out_addr;
    wire [{layer.size-1}:0] layer_{layer.id}_spike_debug;
    wire [15:0] layer_{layer.id}_v_mem_debug [0:{layer.size-1}];
"""
        
        clock_period_ns = 1000.0 / timing.clock_frequency_mhz
        
        return f"""
module snn_top #(
    parameter ADDR_WIDTH = {addr_width},
    parameter CLOCK_PERIOD_NS = {int(clock_period_ns)}
)(
    input wire clk,
    input wire rst,
    
    // External spike input interface
    input wire ext_spike_valid,
    input wire [ADDR_WIDTH-1:0] ext_spike_addr,
    input wire [15:0] ext_spike_data,
    output wire ext_spike_ready,
    
    // External spike output interface  
    output wire ext_spike_out_valid,
    output wire [ADDR_WIDTH-1:0] ext_spike_out_addr,
    output wire [15:0] ext_spike_out_data,
    input wire ext_spike_out_ready,
    
    // Configuration interface
    input wire config_enable,
    input wire [15:0] config_addr,
    input wire [31:0] config_data,
    
    // Debug/monitoring
    output wire [31:0] cycle_counter,
    output wire network_active
);

    // Internal clock and reset
    wire update_enable;
    reg [31:0] cycle_count;
    
    // Update enable generation (could be every cycle or divided)
    assign update_enable = 1'b1;  // Update every clock cycle
    assign cycle_counter = cycle_count;
    
    always @(posedge clk) begin
        if (rst) begin
            cycle_count <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;
        end
    end

{layer_wires}
    
    // Spike router instance
    spike_router router_inst (
        .clk(clk),
        .rst(rst),
        .spike_valid_in(ext_spike_valid),
        .spike_addr_in(ext_spike_addr),
        .spike_timestamp(cycle_count[15:0]),
        .spike_ready(ext_spike_ready),
        .spike_valid_out(ext_spike_out_valid),
        .spike_addr_out(ext_spike_out_addr),
        .spike_weight(ext_spike_out_data),
        .spike_delay(),
        .spike_ready_out(ext_spike_out_ready),
        .config_enable(config_enable),
        .config_addr(config_addr),
        .config_data(config_data)
    );

{layer_instances}
    
    // Network activity monitoring
    wire network_has_spikes;
    assign network_has_spikes = {' | '.join([f'|layer_{layer.id}_spike_debug' for layer in network.layers])};
    assign network_active = network_has_spikes;

endmodule"""
    
    def _generate_xdc_constraints(self, target: FPGATarget, timing: TimingConstraints) -> str:
        """Generate Xilinx Design Constraints (XDC) file."""
        return f"""
# Spiking FPGA Toolchain - Timing Constraints
# Target: {target.target}
# Generated automatically

# Clock constraints
create_clock -period {1000.0 / timing.clock_frequency_mhz:.3f} -name sys_clk [get_ports clk]

# Input/Output constraints
set_input_delay -clock sys_clk {timing.setup_time_ns} [get_ports {{ext_spike_*}}]
set_output_delay -clock sys_clk {timing.clock_to_output_ns} [get_ports {{ext_spike_out_*}}]

# False paths for reset
set_false_path -from [get_ports rst]

# Multicycle paths for configuration
set_multicycle_path -setup 2 -from [get_ports config_*]
set_multicycle_path -hold 1 -from [get_ports config_*]

# Area constraints to encourage clustering
create_pblock pblock_snn_core
resize_pblock pblock_snn_core -add SLICE_X0Y0:SLICE_X50Y50
add_cells_to_pblock pblock_snn_core [get_cells {{*layer_*}}]

# Memory constraints
set_property BLOCK_RAM_PATTERN TDP [get_cells -hier -filter {{NAME =~ *routing_table*}}]
"""
    
    def _generate_sdc_constraints(self, target: FPGATarget, timing: TimingConstraints) -> str:
        """Generate Synopsys Design Constraints (SDC) file for Intel FPGAs."""
        return f"""
# Spiking FPGA Toolchain - Timing Constraints  
# Target: {target.target}
# Generated automatically

# Create clocks
create_clock -period {1000.0 / timing.clock_frequency_mhz:.3f} [get_ports clk]

# Input constraints
set_input_delay -clock clk {timing.setup_time_ns} [get_ports {{ext_spike_*}}]

# Output constraints  
set_output_delay -clock clk {timing.clock_to_output_ns} [get_ports {{ext_spike_out_*}}]

# False paths
set_false_path -from [get_ports rst]

# Multicycle paths
set_multicycle_path -setup 2 -from [get_ports config_*]
"""
    
    def _generate_testbench(self, network: SNNNetwork) -> str:
        """Generate Verilog testbench for simulation."""
        total_neurons = sum(layer.size for layer in network.layers)
        addr_width = max(8, math.ceil(math.log2(total_neurons)))
        
        return f"""
`timescale 1ns / 1ps

module snn_testbench;

    // Testbench signals
    reg clk;
    reg rst;
    reg ext_spike_valid;
    reg [{addr_width-1}:0] ext_spike_addr;
    reg [15:0] ext_spike_data;
    wire ext_spike_ready;
    wire ext_spike_out_valid;
    wire [{addr_width-1}:0] ext_spike_out_addr;
    wire [15:0] ext_spike_out_data;
    reg ext_spike_out_ready;
    reg config_enable;
    reg [15:0] config_addr;
    reg [31:0] config_data;
    wire [31:0] cycle_counter;
    wire network_active;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz clock
    end
    
    // DUT instantiation
    snn_top dut (
        .clk(clk),
        .rst(rst),
        .ext_spike_valid(ext_spike_valid),
        .ext_spike_addr(ext_spike_addr),
        .ext_spike_data(ext_spike_data),
        .ext_spike_ready(ext_spike_ready),
        .ext_spike_out_valid(ext_spike_out_valid),
        .ext_spike_out_addr(ext_spike_out_addr),
        .ext_spike_out_data(ext_spike_out_data),
        .ext_spike_out_ready(ext_spike_out_ready),
        .config_enable(config_enable),
        .config_addr(config_addr),
        .config_data(config_data),
        .cycle_counter(cycle_counter),
        .network_active(network_active)
    );
    
    // Test sequence
    initial begin
        // Initialize
        rst = 1;
        ext_spike_valid = 0;
        ext_spike_addr = 0;
        ext_spike_data = 0;
        ext_spike_out_ready = 1;
        config_enable = 0;
        config_addr = 0;
        config_data = 0;
        
        // Reset sequence
        #50 rst = 0;
        
        // Configure routing table (simplified)
        #100;
        config_enable = 1;
        config_addr = 16'hFFFF;
        config_data = 32'h100; // 256 connections
        #10 config_enable = 0;
        
        // Wait for configuration
        #100;
        
        // Send test spikes
        repeat (10) begin
            ext_spike_valid = 1;
            ext_spike_addr = $random % {total_neurons};
            ext_spike_data = 16'h100; // Fixed weight
            #10 ext_spike_valid = 0;
            #50; // Wait between spikes
        end
        
        // Run simulation
        #10000;
        
        $display("Simulation completed at cycle %d", cycle_counter);
        $display("Network activity: %b", network_active);
        $finish;
    end
    
    // Monitor output spikes
    always @(posedge clk) begin
        if (ext_spike_out_valid && ext_spike_out_ready) begin
            $display("Output spike: addr=%d, data=%d, time=%d", 
                    ext_spike_out_addr, ext_spike_out_data, cycle_counter);
        end
    end
    
    // Dump waveforms
    initial begin
        $dumpfile("snn_simulation.vcd");
        $dumpvars(0, snn_testbench);
    end

endmodule"""
    
    def _generate_stimulus_data(self, network: SNNNetwork) -> List[str]:
        """Generate stimulus data for testbench."""
        stimulus = []
        
        # Generate random spike patterns for input layers
        input_layers = [l for l in network.layers if l.layer_type.value == 'input']
        
        for i in range(1000):  # 1000 time steps
            # Random spike probability
            if i % 10 == 0:  # Spike every 10 cycles
                for layer in input_layers:
                    neuron_id = i % layer.size
                    weight = 256  # Fixed-point 1.0
                    stimulus.append(f"{neuron_id:04x}_{weight:04x}")
            else:
                stimulus.append("0000_0000")
        
        return stimulus