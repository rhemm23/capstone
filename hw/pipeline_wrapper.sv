module pipeline_wrapper
  (
    input clk,
    input rst_n,
    input rdn_load_weights,
    input dnn_load_weights,
    input weight_mem_ready,
    input [63:0] rdn_weight_data [7:0],
    input [63:0] dnn_weight_data [7:0],
    input results_acceptable,
    input csRam1_ext,
    input weRam1_ext,
    input wrAll,
    input [17:0] addrRam1_ext,
    input [7:0]  wrAllData [299:0][299:0],
    input initIpgu,

    output rdn_weights_vld,
    output dnn_weights_vld,
    output rdn_mem_req,
    output dnn_mem_req,
    output [511:0] dnn_results,
    output ipgu_in_rdy
  );

  wire ipgu_out_vld, heu_in_rdy, heu_out_vld, rdn_in_rdy, rdn_out_vld;
  wire iru_in_rdy, iru_out_vld, bcau_in_rdy, bcau_out_vld, dnn_in_rdy, dnn_out_vld;
  wire [7:0] ipgu_q_buffer [4:0][79:0];
  wire [7:0] heu_q_buffer [4:0][79:0];
  wire [7:0] rdn_q_buffer [4:0][79:0];
  wire [7:0] iru_q_buffer [4:0][79:0];
  wire [7:0] bcau_q_buffer [4:0][79:0];
  wire [35:0] angle_data;

  ipgu IPGU (
    .clk(clk),
    .rst_n(rst_n),
    .csRam1_ext(csRam1_ext),
    .weRam1_ext(weRam1_ext),
    .wrAll(wrAll),
    .addrRam1_ext(addrRam1_ext),
    .wrAllData(wrAllData),
    .initIpgu(initIpgu),
    .rdyHeu(heu_in_rdy),

    .rdyIpgu(ipgu_in_rdy),
    .vldIpgu(ipgu_out_vld),
    .ipguOutBufferQ(ipgu_q_buffer));

  heu HEU (
    .clk(clk),
    .rst_n(rst_n),
    .prev_out_ready(ipgu_out_vld),
    .next_in_ready(rdn_in_rdy),
    .d(ipgu_q_buffer),
    
    .in_ready(heu_in_rdy),
    .out_ready(heu_out_vld),
    .q(heu_q_buffer));

  rdn_fp RDN (
    .clk(clk),
    .rst_n(rst_n),
    .load_weights(rdn_load_weights),
    .mem_ready(weight_mem_ready),
    .heu_out_valid(heu_out_vld),
    .iru_in_ready(iru_in_rdy),
    .d(heu_q_buffer),
    .mem_data(rdn_weight_data),
    
    .in_ready(rdn_in_rdy),
    .out_ready(rdn_out_vld),
    .weight_valid(rdn_weights_vld),
    .req_mem(rdn_mem_req),
    .angle_out(angle_data),
    .q(rdn_q_buffer));

  iru IRU (
    .clk(clk),
    .rst_n(rst_n),
    .rnn_out_ready(rdn_out_vld),
    .bcau_in_ready(bcau_in_rdy),
    .rnn_out(angle_data),
    .d(rdn_q_buffer),
    
    .in_ready(iru_in_rdy),
    .out_ready(iru_out_vld),
    .q(iru_q_buffer));

  bcau BCAU (
    .clk(clk),
    .rst_n(rst_n),
    .iru_valid(iru_out_vld),
    .dnn_ready(dnn_in_rdy),
    .iru_results(iru_q_buffer),
    
    .bcau_valid(bcau_out_vld),
    .bcau_ready(bcau_in_rdy),
    .bcau_results(bcau_q_buffer));

  dnn DNN (
    .clk(clk),
    .rst_n(rst_n),
    .mem_ready(weight_mem_ready),
    .load_weights(dnn_load_weights),
    .next_in_ready(results_acceptable),
    .prev_out_ready(bcau_out_vld),
    .d(bcau_q_buffer),
    .mem_data(dnn_weight_data),
    
    .req_mem(dnn_mem_req),
    .in_ready(dnn_in_rdy),
    .out_ready(dnn_out_vld),
    .weight_valid(dnn_weights_vld),
    .results(dnn_results));

endmodule