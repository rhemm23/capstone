
`include "mem_types.vh"
`include "data_types.vh"
`include "platform_if.vh"

module memory
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input t_mem_tx mem_tx,
    input t_if_ccip_Rx rx,

    /*
     * Outputs
     */
    output buffer_addr_valid,
    output t_mem_rx mem_rx,
    output t_if_ccip_Tx tx
  );

  typedef enum reg [2:0] {
    IDLE = 3'b000,
    RD_IMG = 3'b001,
    RD_PRG = 3'b010,
    RD_RNN = 3'b011,
    RD_DNN = 3'b100
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
      mem_rx <= '0;
      tx.c0 <= '0;
      tx.c1 <= '0;
    end else begin
      case (state)
        IDLE: begin
          case (rd_req_type)
            INSTR: 
            RNN_W:
            DNN_W:
            IMAGE:
          endcase
        end
        RD_PRG: begin

        end
      endcase
    end
  end

  assign buffer_addr_valid = |buffer_addr;

endmodule
