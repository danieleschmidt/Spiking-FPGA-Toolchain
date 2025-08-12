// Spike routing network
module spike_router #(
    parameter NUM_NEURONS = 1094
) (
    input wire clk,
    input wire rst,
    
    // Neuron spike inputs
    input wire [NUM_NEURONS-1:0] neuron_spikes,
    
    // External spike interface
    input wire [31:0] spike_in_data,
    input wire spike_in_valid,
    output reg spike_in_ready,
    
    output reg [31:0] spike_out_data,
    output reg spike_out_valid,
    input wire spike_out_ready
);

    // Spike packet format: [timestamp:16][neuron_id:14][valid:1][eop:1]
    localparam TIMESTAMP_WIDTH = 16;
    localparam NEURON_ID_WIDTH = 14;
    
    // Internal spike FIFO
    reg [31:0] spike_fifo [0:255];
    reg [7:0] fifo_write_ptr, fifo_read_ptr;
    reg [8:0] fifo_count;
    
    // Timestamp counter
    reg [TIMESTAMP_WIDTH-1:0] timestamp_counter;
    
    always @(posedge clk) begin
        if (rst) begin
            timestamp_counter <= 0;
            fifo_write_ptr <= 0;
            fifo_read_ptr <= 0;
            fifo_count <= 0;
            spike_out_valid <= 0;
            spike_in_ready <= 1;
        end else begin
            // Increment timestamp
            timestamp_counter <= timestamp_counter + 1;
            
            // Process input spikes
            if (spike_in_valid && spike_in_ready) begin
                spike_fifo[fifo_write_ptr] <= spike_in_data;
                fifo_write_ptr <= fifo_write_ptr + 1;
                fifo_count <= fifo_count + 1;
            end
            
            // Process neuron spikes
            for (integer i = 0; i < NUM_NEURONS; i = i + 1) begin
                if (neuron_spikes[i] && fifo_count < 255) begin
                    spike_fifo[fifo_write_ptr] <= {timestamp_counter, i[NEURON_ID_WIDTH-1:0], 1'b1, 1'b1};
                    fifo_write_ptr <= fifo_write_ptr + 1;
                    fifo_count <= fifo_count + 1;
                end
            end
            
            // Output spike processing
            if (spike_out_ready && fifo_count > 0) begin
                spike_out_data <= spike_fifo[fifo_read_ptr];
                spike_out_valid <= 1;
                fifo_read_ptr <= fifo_read_ptr + 1;
                fifo_count <= fifo_count - 1;
            end else begin
                spike_out_valid <= 0;
            end
            
            // Flow control
            spike_in_ready <= (fifo_count < 240); // Leave some headroom
        end
    end

endmodule