// LIF Neuron #0
module lif_neuron_0 #(
    parameter V_THRESH = 256,
    parameter V_RESET = 0,
    parameter V_REST = 0, 
    parameter DECAY_FACTOR = 243,
    parameter REFRAC_CYCLES = 1
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