module pipeline_wrapper_tb();


logic rdn_load_weights;
logic dnn_load_weights;
logic weight_mem_ready;
logic [63:0] rdn_weight_data [7:0];
logic [63:0] dnn_weight_data [7:0];
logic results_acceptable;
logic csRam1_ext;
logic weRam1_ext;
logic wrAll;
logic [17:0] addrRam1_ext;
logic [7:0]  wrAllData [299:0][299:0];
logic initIpgu;

logic rdn_weights_vld;
logic dnn_weights_vld;
logic rdn_mem_req;
logic dnn_mem_req;
logic [511:0] dnn_results;
logic ipgu_in_rdy;


pipeline_wrapper data_pipeline
  (
    .clk(clk),
    .rst_n(rst_n),
    .rdn_load_weights(rdn_load_weights),
    .dnn_load_weights(dnn_load_weights),
    .weight_mem_ready(weight_mem_ready),
    .rdn_weight_data(rdn_weight_data),
    .dnn_weight_data(dnn_weight_data),
    .results_acceptable(results_acceptable),
    .csRam1_ext(csRam1_ext),
    .weRam1_ext(weRam1_ext),
    .wrAll(wrAll),
    .addrRam1_ext(addrRam1_ext),
    .wrAllData(wrAllData),
    .initIpgu(initIpgu),

    .rdn_weights_vld(rdn_weights_vld),
    .dnn_weights_vld(dnn_weights_vld),
    .rdn_mem_req(rdn_mem_req),
    .dnn_mem_req(dnn_mem_req),
    .dnn_results(dnn_results),
    .ipgu_in_rdy(ipgu_in_rdy)
  );

endmodule