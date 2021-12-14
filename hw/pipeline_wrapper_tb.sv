module pipeline_wrapper_tb();

logic clk;
logic rst_n;
logic rdn_load_weights;
logic dnn_load_weights;
logic weight_mem_ready;
logic [63:0] rdn_weight_data [7:0];
logic [63:0] dnn_weight_data [7:0];
logic results_acceptable;
logic csRam1_ext;
logic weRam1_ext;
logic wrAll;
logic [7:0]  wrAllData [299:0][299:0];
logic initIpgu;

logic rdn_weights_vld;
logic dnn_weights_vld;
logic rdn_mem_req;
logic dnn_mem_req;
logic [511:0] dnn_results;

    wire ipgu_in_rdy;
    reg initIpgu;
    reg rdyHeu;

    logic [7:0] initMemVals [299:0][299:0];
    logic [7:0] wrAllData [299:0][299:0];

    int conversionI = 0;
    int windowNum = 0;
 
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
    .wrAllData(wrAllData),
    .initIpgu(initIpgu),

    .rdn_weights_vld(rdn_weights_vld),
    .dnn_weights_vld(dnn_weights_vld),
    .rdn_mem_req(rdn_mem_req),
    .dnn_mem_req(dnn_mem_req),
    .dnn_results(dnn_results),
    .ipgu_in_rdy(ipgu_in_rdy),
    .dnn_out_vld(dnn_out_vld)
  );

  initial begin
    clk = 0;
    rst_n = 0;
    rdn_load_weights = 0;
    dnn_load_weights = 0;
    weight_mem_ready = 0;
    for (int i = 0; i < 8; i++) rdn_weight_data[i] = 0;
    for (int i = 0; i < 8; i++) dnn_weight_data[i] = 0;
    results_acceptable = '1;
    csRam1_ext = 0;
    weRam1_ext = 0;
    wrAll = 0;
    initIpgu = 0;

    // reset
    @(posedge clk);
    rst_n = 1;
    @(posedge clk);

        rst_n = '0;
        rdyHeu = '0;
        void'(std::randomize(initMemVals));
        @(posedge clk) rst_n = '1;
        data_pipeline.IPGU.ram1.ipgu.mem = initMemVals;
        data_pipeline.IPGU.ram2.mem = '{default:0};
        wrAllData = '{default:0};
        @(posedge clk) initIpgu = '1;
        @(posedge clk) initIpgu = '0;
        fork: hello
            wait(dnn_out_vld) begin $display("Valid dnn"); $stop(); end
            @(posedge ipgu_in_rdy) begin
                $display("Done ipgu sending");
                $stop();
            end
        join
        disable hello; 
            
        $stop();
  end

  always
    #5 clk = ~clk;
endmodule
