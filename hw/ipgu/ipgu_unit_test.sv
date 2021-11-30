module ipgu_unit_test;

reg clk, rst_n;
wire rdyIpgu;
reg initIpgu;
reg rdyHeu;
wire vldIpgu;
wire [7:0] ipguOutBufferQ [79:0];    

    ipgu #(.RAM_DATA_WIDTH(8), .RAM_ADDR_WIDTH($clog2(300)*2)) DUT(   
        .clk,                                         
        .rst_n,                                       
                                                         
        //IPGU <-> ctrlUnit                                  
        .csRam1_ext(1'b0),                                  
        .weRam1_ext(1'b0),                                  
        .addrRam1_ext('0),          
        .wrDataRam1_ext('0),        
        .initIpgu,                                    
        .rdyIpgu,                                     

        //IPGU <-> HEU                                       
        .rdyHeu,                                      
        .vldIpgu,                                 
        .ipguOutBufferQ              
    );                                                       

    
    initial begin
        clk = 0;
        forever #5 clk = !clk; 
    end
    
    initial begin 
        rst_n = '0;
	rdyHeu = '0;
        @(posedge clk) rst_n = '1;
        DUT.ram1.mem = '{default:0};
        @(posedge clk) initIpgu = '1;
        @(posedge clk) initIpgu = '0;
        fork
            forever begin: vldIpguWait
                wait(vldIpgu) begin 
                    if(DUT.state==2)$stop();
                    @(posedge clk) rdyHeu = '1;
                    @(posedge clk) rdyHeu = '0;
                end
            end
            wait(rdyIpgu) disable vldIpguWait;
        join
        wait(vldIpgu) begin
            @(posedge clk) rdyHeu = '1;
            @(posedge clk) rdyHeu = '0;
        end

    end

endmodule
