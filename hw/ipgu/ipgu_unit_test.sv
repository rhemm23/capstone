module ipgu_unit_test;

    reg clk, rst_n;
    wire rdyIpgu;
    reg initIpgu;
    reg rdyHeu;
    wire vldIpgu;
    wire [7:0] ipguOutBufferQ [4:0][79:0];    

    logic [7:0] initMemVals [299:0][299:0];
    logic [7:0] wrAllData [299:0][299:0];

    int conversionI = 0;
    int windowNum = 0;
    int dims[] = {300,240,180,120,60,20,20}; 

    ipgu #(.RAM_DATA_WIDTH(8), .RAM_ADDR_WIDTH($clog2(300)*2)) DUT(   
        .clk,                                         
        .rst_n,                                       
                                                         
        //IPGU <-> ctrlUnit                                  
        .csRam1_ext(1'b0),                                  
        .weRam1_ext(1'b0),                                  
        .initIpgu,                                    
        .rdyIpgu,
	.wrAll('0),
	.wrAllData(wrAllData),                                    

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
        void'(std::randomize(initMemVals));
        @(posedge clk) rst_n = '1;
        DUT.ram1.mem = initMemVals;
        DUT.ram2.mem = '{default:0};
	wrAllData = '{default:0};
        @(posedge clk) initIpgu = '1;
        @(posedge clk) initIpgu = '0;
        fork: hello
            begin 
            forever begin: vldIpguWait
                wait(vldIpgu) begin 
                    if(DUT.ctrlUnit.state==2)$stop();
                    rdyHeu = '1;  void'(checkOutBuffer()); 
                    @(posedge clk) rdyHeu = '0;
		    @(posedge clk);
                end
            end end
            @(posedge rdyIpgu) begin
            $stop();
            end
//            wait(DUT.ram2.addr_x==204&&DUT.ram2.addr_y==0) $stop;
        join_any
        disable hello; 
        repeat(2) begin 
          wait(vldIpgu) begin
              rdyHeu = '1;  void'(checkOutBuffer()); 
              @(posedge clk) rdyHeu = '0;
              @(posedge clk);
          end
        end
        $stop();
    end


    function void checkOutBuffer();
        automatic logic [7:0] rowVals[19:0][299:0] = initMemVals[(windowNum/(dims[conversionI]/10-1))*10+:20];
//        static bit [7:0] windowVals[4:0] = rowVals[19:0][(windowNum%(dims[conversionI]/20))*10+:20];
        automatic logic [7:0] windowVals[4:0][79:0];
	automatic int colsBegin =(windowNum%(dims[conversionI]/10-1))*10;
        for(int i=0; i<5; i++) begin
            for(int j=0; j<4; j++)
                windowVals[i][(j+1)*20-1-:20] = (rowVals[i*4+j][colsBegin+:20]);    
        end

        if(ipguOutBufferQ!=windowVals) begin
            $display("%p\n\n%p \n\n %d,%d\n\n", ipguOutBufferQ[0][19:0], windowVals[0][19:0], (windowNum/(dims[conversionI]/20))*10,colsBegin);
            $display("Tests failed for windowNum %d conversionI %d converting from %d to %d",windowNum, conversionI, dims[conversionI], dims[conversionI+1]);
            $stop(); 
        end
	else
		$display("Pass %p %p",ipguOutBufferQ,windowVals);
        windowNum++;
        if(windowNum==(dims[conversionI]/10-1)*(dims[conversionI]/10-1)) begin
            logic [7:0] tempMem [299:0][299:0];
            tempMem = '{default:0}; 
            conversionI++;
            windowNum = 0;
            for (int i=0;i<dims[conversionI-1];i++) 
	    for (int j=0;j<dims[conversionI-1];j++) begin
                tempMem[i*dims[conversionI]/dims[conversionI-1]][j*dims[conversionI]/dims[conversionI-1]] = initMemVals[i][j];
	    end
            initMemVals = tempMem; 
        end
    endfunction
endmodule
