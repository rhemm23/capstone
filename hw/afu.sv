
`include "platform_if.vh"

import data_types::*;

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

  t_mem_rx mem_rx;
  t_mem_tx mem_tx;

  wire buffer_addr_valid;

  memory mem (
    .clk(clk),
    .rst_n(rst_n),
    .mem_tx(mem_tx),
    .rx(rx),
    .buffer_addr_valid(buffer_addr_valid),
    .mem_rx(mem_rx),
    .tx(tx)
  );

  cpu cpu (
    .clk(clk),
    .rst_n(rst_n),
    .buffer_addr_valid(buffer_addr_valid),
    .mem_rx(mem_rx),
    .mem_tx(mem_tx)
  );

endmodule
