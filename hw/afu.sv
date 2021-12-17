
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

  wire image_data_ready;
  wire [7:0]  pixel_data [63:0];
  wire initIpgu;
  wire rdyIpgu;

  wire rdnReqWeightMem;
  wire doneWeightRdn;
  wire [63:0]  rdn_weights [7:0];
  wire begin_rdn_load;
  wire weights_ready;

  wire dnnResVld;
  wire [511:0] dnnResults;
  wire dnnResRdy;
  wire begin_dnn_load;

  wire dnnReqWeightMem;
  wire doneWeightDnn;
  wire [63:0]  dnn_weights [7:0];

  wire ipgu_req_mem;

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

  control_wrapper top_level_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .buffer_addr_valid(buffer_addr_valid),
    .data_valid(data_valid),
    .write_done(write_done),
    .read_data(read_data),
    .rdyIpgu(rdyIpgu),
    .ipgu_req_mem(ipgu_req_mem),

    .address(address),
    .write_data(write_data),
    .read_request_valid(read_request_valid),
    .write_request_valid(write_request_valid),

    .image_data_ready(image_data_ready),
    .ipgu_data(pixel_data),
    .initIpgu(initIpgu),

    .rdnReqWeightMem(rdnReqWeightMem),
    .doneWeightRdn(doneWeightRdn),
    .rdn_weights(rdn_weights),
    .begin_rdn_load(begin_rdn_load),
    .weights_ready(weights_ready),

    .dnnResVld(dnnResVld),
    .dnnResults(dnnResults),
    .dnnResRdy(dnnResRdy),
    .begin_dnn_load(begin_dnn_load),

    .dnnReqWeightMem(dnnReqWeightMem),
    .doneWeightDnn(doneWeightDnn),
    .dnn_weights(dnn_weights)
  );



  pipeline_wrapper data_pipeline
  (
    .clk(clk),
    .rst_n(rst_n),
    .rdn_load_weights(begin_rdn_load),
    .dnn_load_weights(begin_dnn_load),
    .weight_mem_ready(weights_ready),
    .rdn_weight_data(rdn_weights),
    .dnn_weight_data(dnn_weights),
    .results_acceptable(dnnResRdy),
    .image_data_ready(image_data_ready),
    .pixel_data(pixel_data),
    .initIpgu(initIpgu),

    .rdn_weights_vld(doneWeightRdn),
    .dnn_weights_vld(doneWeightDnn),
    .ipgu_req_mem(ipgu_req_mem),
    .rdn_mem_req(rdnReqWeightMem),
    .dnn_mem_req(dnnReqWeightMem),
    .dnn_results(dnnResults),
    .ipgu_in_rdy(rdyIpgu),
    .dnn_out_vld(dnnResVld)
  );

endmodule
