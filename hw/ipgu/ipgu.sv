module ipgu #(RAM_DATA_WIDTH = 8, RAM_ADDR_WIDTH = 18) 
(   input   clk,
    input   rst_n,

    //IPGU <-> ctrlUnit
    input   csRam1_ext,
    input   weRam1_ext,
    input   wrAll,
    input   [7:0]     wrAllData [300-1:0][300-1:0],
    input   initIpgu,
    output  rdyIpgu,
    
    //IPGU <-> HEU
    input   rdyHeu, 
    output  reg vldIpgu,
    output  reg [7:0] ipguOutBufferQ [4:0][79:0]
);

        
    wire [RAM_ADDR_WIDTH-1:0] addrRam1, addrRam1_int;
    wire [($clog2(240)*2)-1:0] addrRam2;
    wire [RAM_DATA_WIDTH-1:0] wrDataRam1, rdDataRam1, rdDataRam2; //rdDataRam2 == wrDataRam1_int

    wire weRam1, weRam1_int, weRam2;
    wire csRam1_int, csRam2_int;

    wire [RAM_ADDR_WIDTH/2-1:0] addrX, addrY;
    wire [$clog2(240)-1:0] scaledX, scaledY;


    //Ram1 Signals: Choose who is in control of RAM1 based on csRam1_ext
    assign addrRam1     = addrRam1_int;
    assign wrDataRam1   = rdDataRam2;
    assign weRam1       = csRam1_ext ? weRam1_ext       : weRam1_int;


    //Choose whether RAM address is scaled or normal for each RAM
    assign addrRam1_int = weRam1_int ? {1'b0,scaledY,1'b0, scaledX} : {addrY, addrX};
    assign addrRam2 = weRam2 ? {scaledY,scaledX} : {addrY[$clog2(240)-1:0],addrX[$clog2(240)-1:0]};

    
    /////////////////////////////////////////////////////
    // RAM instances    
    /////////////////////////////////////////////////////

    ram_wrAll #(.DATA_WIDTH(RAM_DATA_WIDTH)) ram1 (
        .clk, .addr(addrRam1), .rdData(rdDataRam1), .wrData(wrDataRam1), 
        .cs(csRam1_ext|csRam1_int), .we(weRam1), .wrAllData, .wrAll
    );
    
    ram #(.DEPTH_X(240), .DEPTH_Y(240), .DATA_WIDTH(RAM_DATA_WIDTH)) ram2 (
        .clk, .addr(addrRam2), .rdData(rdDataRam2), .wrData(rdDataRam1), 
        .cs(csRam2_int), .we(weRam2)
    );

    /////////////////////////////////////////////////////
    // IPGU Out Buffer    
    /////////////////////////////////////////////////////

    logic [7:0] ipguOutQ [399:0];
    
    out_fifo #(.Q_DEPTH(400)) ipgu_out(.clk,.rst_n,    
                        .en( (csRam1_int&weRam1_int) | (csRam2_int&weRam2)), //if either RAM is being internally written to, write to buffer as well
                        .d( (csRam1_int&weRam1_int) ? rdDataRam2 : rdDataRam1),
                        .q({<<8{ipguOutQ}}));  
     
    //https://www.amiq.com/consulting/2017/05/29/how-to-pack-data-using-systemverilog-streaming-operators/#reverse_bits
    generate for(genvar i=0; i<5; i++) begin
        assign ipguOutBufferQ[i] = ipguOutQ[(i+1)*80-1-:80];     
    end endgenerate

    /////////////////////////////////////////////////////
    // Interconnect Signals
    /////////////////////////////////////////////////////

    wire windowDone;
    wire incX;
    wire [3:0] numWindows;
    wire [2:0] convertI;

    wire [RAM_ADDR_WIDTH/2-1:0] addrXBegin, addrXEnd, addrYBegin, addrYEnd;
    
    //Address Compute Unit: auto-args hook to ports
    ipgu_addr_compute #(.RAM_ADDR_WIDTH(RAM_ADDR_WIDTH)) addrCompute (.*);

    //IPGU Control Unit: auto-args hook to ports
    ipgu_ctrl_unit #(.RAM_ADDR_WIDTH(RAM_ADDR_WIDTH)) ctrlUnit (.*);

    
endmodule
