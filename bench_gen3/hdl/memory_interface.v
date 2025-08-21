// Memory interface for synaptic weights
module memory_interface #(
    parameter NUM_SYNAPSES = 219
) (
    input wire clk,
    input wire rst,
    
    // AXI4-Lite interface
    input wire [31:0] s_axi_awaddr,
    input wire s_axi_awvalid,
    output reg s_axi_awready,
    input wire [31:0] s_axi_wdata,
    input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid,
    output reg s_axi_wready,
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,
    
    input wire [31:0] s_axi_araddr,
    input wire s_axi_arvalid,
    output reg s_axi_arready,
    output reg [31:0] s_axi_rdata,
    output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid,
    input wire s_axi_rready
);

    // Synaptic weight memory (16-bit weights)
    reg [15:0] synapse_weights [0:NUM_SYNAPSES-1];
    
    // AXI4-Lite state machine
    typedef enum logic [2:0] {
        IDLE,
        WRITE_ADDR,
        WRITE_DATA,
        WRITE_RESP,
        READ_ADDR,
        READ_DATA
    } axi_state_t;
    
    axi_state_t current_state, next_state;
    
    // Address decoding
    wire [31:0] word_addr = s_axi_awaddr[31:2]; // Word-aligned addresses
    wire [31:0] read_word_addr = s_axi_araddr[31:2];
    
    always @(posedge clk) begin
        if (rst) begin
            current_state <= IDLE;
            s_axi_awready <= 0;
            s_axi_wready <= 0;
            s_axi_bvalid <= 0;
            s_axi_arready <= 0;
            s_axi_rvalid <= 0;
        end else begin
            case (current_state)
                IDLE: begin
                    if (s_axi_awvalid) begin
                        current_state <= WRITE_ADDR;
                        s_axi_awready <= 1;
                    end else if (s_axi_arvalid) begin
                        current_state <= READ_ADDR;
                        s_axi_arready <= 1;
                    end
                end
                
                WRITE_ADDR: begin
                    s_axi_awready <= 0;
                    if (s_axi_wvalid) begin
                        current_state <= WRITE_DATA;
                        s_axi_wready <= 1;
                        // Write data to memory
                        if (word_addr < NUM_SYNAPSES) begin
                            synapse_weights[word_addr] <= s_axi_wdata[15:0];
                        end
                    end
                end
                
                WRITE_DATA: begin
                    s_axi_wready <= 0;
                    current_state <= WRITE_RESP;
                    s_axi_bvalid <= 1;
                    s_axi_bresp <= 2'b00; // OKAY response
                end
                
                WRITE_RESP: begin
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 0;
                        current_state <= IDLE;
                    end
                end
                
                READ_ADDR: begin
                    s_axi_arready <= 0;
                    current_state <= READ_DATA;
                    s_axi_rvalid <= 1;
                    // Read data from memory
                    if (read_word_addr < NUM_SYNAPSES) begin
                        s_axi_rdata <= {16'b0, synapse_weights[read_word_addr]};
                        s_axi_rresp <= 2'b00; // OKAY response
                    end else begin
                        s_axi_rdata <= 32'b0;
                        s_axi_rresp <= 2'b10; // SLVERR response
                    end
                end
                
                READ_DATA: begin
                    if (s_axi_rready) begin
                        s_axi_rvalid <= 0;
                        current_state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule