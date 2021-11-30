
`include "platform_if.vh"

module memory
import data_types::*;
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

  localparam PROGRAM_PAGES = 256;

  typedef enum logic [2:0] {
    IDLE = 3'b000,
    RD_IMG = 3'b001,
    RD_PRG = 3'b010,
    RD_RNN = 3'b011,
    RD_DNN = 3'b100,
    DONE = 3'b101
  } mem_state;

  mem_state state;

  reg [31:0] addr;
  reg [31:0] sent_cnt;
  reg [31:0] recv_cnt;

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
      recv_cnt <= '0;
      sent_cnt <= '0;
      mem_rx <= '0;
      tx.c0 <= '0;
      tx.c1 <= '0;
      addr <= '0;
    end else begin
      case (state)
        IDLE: begin
          case (mem_tx.req_type)
            IMAGE: state <= RD_IMG;
            INSTR: state <= RD_PRG;
            RNN_W: state <= RD_RNN;
            DNN_W: state <= RD_DNN;
          endcase
          if (mem_tx.req_type != NONE) begin
            sent_cnt <= '0;
            recv_cnt <= '0;
            addr <= mem_tx.addr;
          end
        end
        RD_PRG: begin
          // Send read request if not almost full
          if (sent_cnt < PROGRAM_PAGES && !rx.c0TxAlmFull) begin
            tx.c0.hdr <= '{
              eVC_VA,
              2'b00,
              eCL_LEN_1,
              eREQ_RDLINE_I,
              6'h00,
              t_ccip_clAddr'(buffer_addr + addr + sent_cnt),
              sent_cnt
            };
            sent_cnt <= sent_cnt + 1;
            tx.c0.valid <= 1;
          end else begin
            tx.c0.valid <= 0;
          end

          // Check for read responses
          if (rx.c0.rspValid && rx.c0.hdr.resp_type == eRSP_RDLINE) begin
            mem_rx.cpu_program[(rx.c0.hdr.mdata * 512) +: 512] = rx.c0.data;
            recv_cnt <= recv_cnt + 1;
          end

          // Received all pages
          if (recv_cnt == PROGRAM_PAGES) begin
            mem_rx.status <= INSTR_VALID;
            state <= DONE;
          end
        end
        DONE: begin
          mem_rx.status <= NONE_VALID;
          state <= IDLE;
        end
      endcase
    end
  end

  assign buffer_addr_valid = |buffer_addr;

endmodule
