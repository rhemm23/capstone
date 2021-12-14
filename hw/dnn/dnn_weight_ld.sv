module dnn_weight_ld
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input mem_ready,
    input [63:0] mem_data [7:0],

    /*
     * Outputs
     */
    output [63:0] weight_bus,
    output [2:0] layer_sel,
    output [3:0] neuron_sel,
    output [6:0] weight_sel,
    output write_weight,
    output weight_valid,
    output req_mem
  );

  typedef enum logic [2:0] {
    IDLE,
    LD_A1,
    LD_A2,
    LD_A3
  } ld_state;

  ld_state state;

  reg [2:0] word_cnt;
  reg [3:0] neuron_cnt;

  reg [511:0] mem_page;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          neuron_cnt <= 4'b0000;
          state <= REQ_PAGE;
        end
        REQ_PAGE: begin
          
        end
      endcase
    end
  end

endmodule
