
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

  wire [31:0] address;
  wire [511:0] data;

  wire data_valid;
  wire request_valid;
  wire buffer_addr_valid;

  memory mem (
    .clk(clk),
    .rst_n(rst_n),
    .request_valid(request_valid),
    .address(address),
    .rx(rx),
    .buffer_addr_valid(buffer_addr_valid),
    .data_valid(data_valid),
    .data(data),
    .tx(tx)
  );

  ctrl_unit ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .buffer_addr_valid(buffer_addr_valid),
    .data_valid(data_valid),
    .data(data),
    .address(address),
    .request_valid(request_valid)
  );

endmodule
