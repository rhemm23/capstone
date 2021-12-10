
`include "platform_if.vh"

module memory
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input request_valid,
    input [31:0] address,
    input t_if_ccip_Rx rx,

    /*
     * Outputs
     */
    output buffer_addr_valid,
    output data_valid,
    output [511:0] data,
    output t_if_ccip_Tx tx
  );

  typedef enum logic [1:0] {
    IDLE = 2'b00,
    WAIT = 2'b01
  } mem_state;

  mem_state state;

  wire [63:0] buffer_addr;

  csrs ctrl_regs (
    .clk(clk),
    .rst_n(rst_n),
    .rx(rx.c0),
    .buffer_addr(buffer_addr),
    .tx(tx.c2)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      tx.c0 <= '0;
      tx.c1 <= '0;
    end else begin
      case (state)
        IDLE: begin
          if (request_valid) begin
            tx.c0.hdr <= '{
              eVC_VA,
              2'b00,
              eCL_LEN_1,
              eREQ_RDLINE_I,
              6'h00,
              t_ccip_clAddr'(buffer_addr + address),
              16'h0000
            };
            tx.c0.valid <= 1;
            state <= WAIT;
          end else begin
            tx.c0.valid <= 0;
          end
        end
        WAIT: begin if (rx.c0.rspValid && rx.c0.hdr.resp_type == eRSP_RDLINE && rx.c0.hdr.mdata == 16'h0000) begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign data_valid = (rx.c0.rspValid) &&
                      (rx.c0.hdr.resp_type == eRSP_RDLINE) &&
                      (rx.c0.hdr.mdata == 16'h0000) &&
                      (state == WAIT);

  assign data = rx.c0.data;
  assign buffer_addr_valid = |buffer_addr;

endmodule
