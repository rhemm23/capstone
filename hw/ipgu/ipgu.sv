module ipgu #(RAM_DATA_WIDTH = 8, RAM_ADDR_WIDTH = 18) 
(   input   clk,
    input   rst_n,

    //IPGU <-> ctrlUnit
    input   csRam1_ext,
    input   weRam1_ext,
    input   [RAM_ADDR_WIDTH-1:0]  addrRam1_ext,
    input   [RAM_DATA_WIDTH-1:0]  wrDataRam1_ext,
    input   initIpgu,
    output  rdyIpgu,
    
    //IPGU <-> HEU
    input   rdyHeu, 
    output  reg vldIpgu,
    output  reg [7:0] ipguOutBufferQ [4:0][79:0]
);

    //Scaling luts //can probably create scaling reg which optimizes scaling
    reg [3:0] numWindows;
    reg [(9+6)-1:0] scaleFactor;
    reg [2:0] convertI;

    always_comb begin
        case(convertI) 
            'd0: begin numWindows <= 15; scaleFactor <= {'0,6'b110011}; end
            'd1: begin numWindows <= 12; scaleFactor <= {'0,6'b110000}; end
            'd2: begin numWindows <= 9; scaleFactor <= {'0,6'b101010}; end
            'd3: begin numWindows <= 6; scaleFactor <= {'0,6'b100000}; end
            'd4: begin numWindows <= 3; scaleFactor <= {'0,6'b010101}; end
            'd5: begin numWindows <= 1; scaleFactor <= {9'b1,'0}; end
            default: begin numWindows <= 0; scaleFactor <= 0; end
         endcase
    end
    
    logic [RAM_ADDR_WIDTH-1:0] addrRam1, addrRam1_int;
    logic [($clog2(240)*2)-1:0]  addrRam2;
    assign addrRam1 = csRam1_ext?addrRam1_ext:addrRam1_int;

    logic [RAM_DATA_WIDTH-1:0] wrDataRam1, rdDataRam1, rdDataRam2; //rdDataRam2 == wrDataRam1_int
    assign wrDataRam1 = csRam1_ext?wrDataRam1_ext:rdDataRam2;

    logic weRam1, weRam1_int, weRam2;
    assign weRam1 = csRam1_ext?weRam1_ext:weRam1_int;

    logic csRam1, csRam1_int, csRam2, csRam2_int, csRam1_d1, csRam2_d1;

    /////////////////////////////////////////////////////
    //more internal signals
    /////////////////////////////////////////////////////
    wire windowDone;

    reg convertDone;
    reg [RAM_ADDR_WIDTH/2-1:0] addrXBegin, addrXEnd, addrYBegin, addrYEnd;
    reg [RAM_ADDR_WIDTH/2-1:0] addrX, addrY;
    reg [$clog2(240)-1:0] scaledX, scaledY;
    wire [$clog2(240)-1:0] lutAddrX, lutAddrY;

    wire [6-1:0] wastedDecimalPlacesX, wastedDecimalPlacesY;
    
    ipgu_mult_lut lutX(.scaleNum(convertI), .addr(addrX), .addrOut(lutAddrX));
    ipgu_mult_lut lutY(.scaleNum(convertI), .addr(addrY), .addrOut(lutAddrY));


    //assign scaledAddr = {(addrY*nextNumWindows)/numWindows, (addrX*nextNumWindows)/numWindows};
    
    assign addrRam1_int = weRam1_int ? {1'b0,scaledY,1'b0, scaledX} : {addrY, addrX};
    assign addrRam2 = weRam2 ? {scaledY,scaledX} : {addrY[$clog2(240)-1:0],addrX[$clog2(240)-1:0]};

    assign windowDone = addrYEnd==addrY && addrXEnd==addrX+1; //last pixel of a windows will be read next 

    assign csRam1_int = csRam1|csRam2_d1;   
    assign csRam2_int = csRam2|csRam1_d1;

    reg incX, incConvertI, resetConvertI, clrConvertDone;
    
    /////////////////////////////////////////////////////

    ram #(.DATA_WIDTH(RAM_DATA_WIDTH)) ram1 (
        .clk, .addr(addrRam1), .rdData(rdDataRam1), .wrData(wrDataRam1), 
        .cs(csRam1_ext|csRam1_int), .we(weRam1)
    );
    
    ram #(.DEPTH_X(240), .DEPTH_Y(240), .DATA_WIDTH(RAM_DATA_WIDTH)) ram2 (
        .clk, .addr(addrRam2), .rdData(rdDataRam2), .wrData(rdDataRam1), 
        .cs(csRam2_int), .we(weRam2)
    );

    logic [7:0] ipguOutQFlipped [399:0];
    
    out_fifo #(.Q_DEPTH(400)) ipgu_out(.clk,.rst_n,    
                        .en((csRam1_int&weRam1_int)|(csRam2_int&weRam2)), //if either RAM is being internally written to, write to buffer as well
                        .d((csRam1_int&weRam1_int)?rdDataRam2:rdDataRam1),
                        .q({<<8{ipguOutQFlipped}}));  
     
/*    generate 
        for(genvar i=0; i<80; i++)
            
    endgenerate*/
    //https://www.amiq.com/consulting/2017/05/29/how-to-pack-data-using-systemverilog-streaming-operators/#reverse_bits
    generate 
    for(genvar i=0; i<5; i++) begin
        assign ipguOutBufferQ[i] = ipguOutQFlipped[(i+1)*80-1-:80]; //potential issue    
    end
    endgenerate
    //weRam1_int and weRam2
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            weRam1_int <= '0;
            weRam2 <= '0;
            csRam1_d1 <= '0; //similar to above
            csRam2_d1 <= '0;
        end
        else begin
            weRam1_int <= csRam2&!weRam2; //if reading from Ram2, write to Ram1 in next cycle
            weRam2 <= csRam1&!weRam1_int; //if reading from Ram1, write to Ram2 in next cycle
            csRam1_d1 <= csRam1; //similar to above
            csRam2_d1 <= csRam2;
        end
    end
 
    ///////////////////////////////////////////
    // SM states //
    ///////////////////////////////////////////
    typedef enum reg [2:0] {IDLE, RAM1_SRC, COMPLETE_LAST_WRITE, RAM2_SRC, WAIT_HEU_RDY} state_t;    
    state_t state,nxt_state;


    
    //// Infer state register next ////
    always_ff @(posedge clk, negedge rst_n)
        if (!rst_n)
            state <= IDLE;
        else
            state <= nxt_state;


    //AddrX Counter
    always_ff @(posedge clk, negedge rst_n)
        if(!rst_n)
            addrX <= '0;
        else begin
            if(incX) 
                addrX <= (addrX==addrXEnd) ? addrXBegin : addrX+1;
        end

    //AddrY Counter
    always_ff @(posedge clk, negedge rst_n)
        if (!rst_n)
            addrY <= '0;
        else begin
            if(incX && (addrX==addrXEnd))
                addrY <= (addrY==addrYEnd) ? addrYBegin : addrY+1;
        end 

    //addrYBegin
    always_ff @(posedge clk, negedge rst_n)
        if(!rst_n)
            addrYBegin <= '0;
        else begin
            if(windowDone && addrXBegin+20==numWindows*20) begin //last pixel of a row of windows be read next
                if(addrYBegin+20==numWindows*20)
                    addrYBegin <= '0;
                else
                    addrYBegin <= addrYBegin+10;
            end
        end

    //addrXBegin
    always_ff @(posedge clk, negedge rst_n)
        if(!rst_n)
            addrXBegin <= '0;
        else begin
            if(windowDone) begin //last pixel of a windows will be read next
                if(addrXBegin+20==numWindows*20)
                    addrXBegin <= '0;
                else
                    addrXBegin <= addrXBegin+10;
            end
        end
    
    //addrXEnd and addrYEnd
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            addrXEnd <= 19;
            addrYEnd <= 19;
        end
        else begin
            addrXEnd <= addrXBegin+20-1;
            addrYEnd <= addrYBegin+20-1;
        end
    end
        
    //convertI
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            convertI <= '0;
        else begin
            if(resetConvertI)
                convertI <= '0;
            else begin 
                if (incConvertI)
                    convertI <= convertI+1;
            end
        end
    end

    //convertDone
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            convertDone <= '0;
        else begin
            if(clrConvertDone)
                convertDone <= '0;
            else if(windowDone && addrXBegin+20==numWindows*20 && addrYBegin+20==numWindows*20)
                convertDone <= 1'b1;
        end
    
    end

    //scaledAddr lut output transferred to scaled Addr with 1 cycle delay
    always_ff @(posedge clk) begin
        scaledX <= lutAddrX;
        scaledY <= lutAddrY;
    end

    assign rdyIpgu = convertDone&&convertI==4;//transfer from 60x60 to 20x20 done
    

    //////////////////////////////////////
    // Implement state tranisiton logic //
    /////////////////////////////////////
    always_comb
        begin
            //////////////////////
            // Default outputs //
            ////////////////////
            nxt_state = state;    
            csRam1 = 1'b0;
            csRam2 = 1'b0;
            incX = '0;
            incConvertI = '0;            
            resetConvertI = '0;    
            vldIpgu = '0;
            clrConvertDone = '0;
            case (state)
                IDLE : begin
                    if(initIpgu) begin
                        nxt_state = RAM1_SRC;
                        incX = '1;
                        csRam1 = '1; 
                        resetConvertI = '1;
                    end
                end 
                RAM1_SRC: begin
                    csRam1 = '1;
                    incX = '1;
                    if(addrX==addrXEnd && addrY==addrYEnd) begin
                        nxt_state = COMPLETE_LAST_WRITE;
                    end
                end
                COMPLETE_LAST_WRITE: begin
        	    nxt_state = WAIT_HEU_RDY;
                end
                WAIT_HEU_RDY: begin
                    vldIpgu = '1;
                    if(rdyHeu) begin
                        if(convertDone) begin
                            incConvertI = '1;
                            clrConvertDone = '1;
                            if(convertI==5)
                                nxt_state = IDLE;
                            else begin
                                if(convertI[0])
                                    nxt_state = RAM1_SRC;
                                else
                                    nxt_state = RAM2_SRC;
                            end
                        end
                        else begin
                            incX = '1;
                            if(convertI[0]) begin
                                nxt_state = RAM2_SRC;
                                csRam2 = 1'b1;
                            end
                            else begin
                                nxt_state = RAM1_SRC;
                                csRam1 = 1'b1;
                            end
                        end
                    end
                end
                RAM2_SRC: begin
                     csRam2 = '1; 
                     incX = '1;
                     if(addrX==addrXEnd && addrY==addrYEnd) begin
                         nxt_state = COMPLETE_LAST_WRITE;
                     end
                end
                default: begin
                     nxt_state = IDLE;
                end
                //don't need a default case since all four states are used 
            endcase
        end

endmodule
