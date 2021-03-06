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

    logic [7:0] initMemVals [299:0][299:0];

    int conversionI = 0;
    int windowNum = 0;
    int fd, windowCnt; 
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

    fd = $fopen("/filespace/s/sjain75/ece554/capstone/hw/ipgu/demo_image.bin","rb");
    if(fd)
        $display("Reading Image");
    else begin
        $display("Failed");
        $stop();
    end
    for(int i=0;i<300;i++)
        for(int j=0;j<300;j++)
            $fread(wrAllData[i][j],fd);

    $fclose(fd);


    rdn_load_weights = 0;
    dnn_load_weights = 0;
    weight_mem_ready = 0;

    // reset
    @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    rdn_load_weights = '1;
    dnn_load_weights = '1;
    @(posedge clk);
    dnn_load_weights = '0;
    rdn_load_weights = '0;
    weight_mem_ready = '1;

    fork
	wait(rdn_weights_vld)  $stop(); 
	wait(dnn_weights_vld)  $stop(); 
    join
    rdn_load_weights = '0;
    dnn_load_weights = '0;
    weight_mem_ready = '0;
    $display("done weights");
    
    @(posedge clk) wrAll = '1;
    @(posedge clk) wrAll = '0;
    @(posedge clk) initIpgu = '1;
    @(posedge clk) initIpgu = '0;

    fork: hello
	forever begin: logSentForever
		wait(data_pipeline.IPGU.rdyHeu&&data_pipeline.IPGU.vldIpgu) begin $display("Window %d sent by IPGU", windowCnt); windowCnt++; 
@(posedge clk); @(posedge clk); end
       	end
        repeat(4) wait(dnn_out_vld) begin $display("Valid dnn"); $stop(); end		
	@(posedge ipgu_in_rdy) begin
	    disable logSentForever;
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
