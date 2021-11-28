
`include "platform_if.vh"

module afu
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input t_if_ccip_Rx rx,

    /*
     * Outputs
     */
    output t_if_ccip_Tx tx
  );

  typedef enum reg [1:0] {
    WAIT_BUF = 2'b00,
    FCH_INST = 2'b01,
    EXECUTE = 2'b10,
    DONE = 2'b11
  } afu_state;

  afu_state state;

  wire [511:0] mem_data;

  wire buf_addr_valid;
  wire mem_data_valid;

  memory mem (
    .clk(clk),
    .rst_n(rst_n),
    .rx(rx),
    .buf_addr_valid(buf_addr_valid),
    .mem_data_valid(mem_data_valid),
    .mem_data(mem_data),
    .tx(tx)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= WAIT_BUF;
    end else begin
      case (state)
        WAIT_BUF: if (buf_addr_valid) begin
          state <= FCH_INST;
        end
        FCH_INST: begin
          /* TODO */
        end
        EXECUTE: begin
          /* TODO */
        end
        DONE: begin
          /* TODO */
        end
      endcase
    end
  end

endmodule
