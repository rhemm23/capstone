module cpu
import data_types::*;
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input buffer_addr_valid,
    input t_mem_rx mem_rx,

    /*
     * Outputs
     */
    output t_mem_tx mem_tx
  );

  typedef enum logic [2:0] {
    WAIT_BUF = 3'b000,
    REQ_PROG = 3'b001,
    WAIT_PROG = 3'b010,
    EXECUTING = 3'b011,
    DONE = 3'b100
  } cpu_state;

  cpu_state state;
  t_program program;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= WAIT_BUF;
      program <= '0;
      mem_tx <= '0;
    end else begin
      case (state)
        WAIT_BUF: if (buffer_addr_valid) begin
          mem_tx <= { INSTR, '0 };
          state <= REQ_PROG;
        end
        REQ_PROG: begin
          mem_tx <= '0;
          state <= WAIT_PROG;
        end
        WAIT_PROG: if (mem_rx.status == INSTR_VALID) begin
          state <= EXECUTING;
          program <= mem_rx.cpu_program;
        end
        EXECUTING: begin
          /* TODO */
        end
      endcase
    end
  end

endmodule
