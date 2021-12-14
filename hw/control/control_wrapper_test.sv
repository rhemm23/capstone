module control_wrapper_test;

bit clk;                                                   
bit rst_n;                                                 
                                                             
//controlWrapper(ctrl_unit) <-> memory                       
bit buffer_addr_valid;                                     
bit data_valid;                                            
bit write_done;                                            
bit [511:0] read_data;                                     
wire [31:0] address;                                       
wire [511:0] write_data;                                   
wire read_request_valid;                                   
wire write_request_valid;                                  
                                                             
//fdp(IPGU) <-> controlWrapper(ctrlUnit)                     
wire            wrAll;                                     
wire   [8-1:0]  wrAllData [300-1:0][300-1:0];              
wire            initIpgu;                                  
bit             rdyIpgu;                                   
                                                             
//rdn <-> ctrl_unit                                          
//weights                                                    
bit           rdnReqWeightMem;                             
bit           doneWeightRdn;                               
wire  [63:0]  rdn_weights [7:0];                           
                                                             
//dnn <-> ctrl_unit                                          
bit           dnnResVld;                                   
bit   [1085:0]dnnResults;                                  
wire          dnnResRdy;                                   
//weights                                                    
bit           dnnReqWeightMem;                             
bit           doneWeightDnn;                               
wire  [63:0]  dnn_weights [7:0];                         
                                                             
control_wrapper controlWrapper(.*);

initial begin
    clk = 0;
    forever #5 clk = !clk; 
end

initial begin 
    rst_n = '0;
    @(posedge clk) rst_n='1;
    @(posedge clk); 
end
endmodule
