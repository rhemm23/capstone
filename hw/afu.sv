
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
  wire [511:0] read_data;
  wire [511:0] write_data;

  wire data_valid;
  wire write_done;
  wire read_request_valid;
  wire write_request_valid;
  wire buffer_addr_valid;

  memory mem (
    .clk(clk),
    .rst_n(rst_n),
    .read_request_valid(read_request_valid),
    .write_request_valid(write_request_valid),
    .address(address),
    .data_d(write_data),
    .rx(rx),
    .buffer_addr_valid(buffer_addr_valid),
    .data_valid(data_valid),
    .write_done(write_done),
    .data_q(read_data),
    .tx(tx)
  );

  ctrl_unit ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .buffer_addr_valid(buffer_addr_valid),
    .data_valid(data_valid),
    .write_done(write_done),
    .read_data(read_data),
    .address(address),
    .write_data(write_data),
    .read_request_valid(read_request_valid),
    .write_request_valid(write_request_valid)
  );

  pipeline_wrapper data_pipeline
  (
    .clk(clk),
    .rst_n(rst_n),
    .rdn_load_weights(),
    .dnn_load_weights(),
    .weight_mem_ready(),
    .rdn_weight_data(),
    .dnn_weight_data(),
    .results_acceptable(),
    .csRam1_ext(),
    .weRam1_ext(),
    .wrAll(),
    .addrRam1_ext(),
    .wrAllData(),
    .initIpgu(),

    .rdn_weights_vld(),
    .dnn_weights_vld(),
    .rdn_mem_req(),
    .dnn_mem_req(),
    .dnn_results(),
    .ipgu_in_rdy()
  );

endmodule
