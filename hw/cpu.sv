
`include "mem_types.vh"

module cpu
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

  always_ff (@posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mem_tx <= '0;
    end
  end

endmodule