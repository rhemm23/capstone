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
    output  reg [7:0] ipguOutBufferQ [79:0]
);

    //Scaling luts //can probably create scaling reg which optimizes scaling
    reg [3:0] numWindows;
    reg [3:0] nextNumWindows;
    reg [2:0] convertI;

    always_comb begin
        case(convertI) 
            'd0: begin numWindows <= 15; nextNumWindows <= 12; end
            'd1: begin numWindows <= 12; nextNumWindows <= 9; end
            'd2: begin numWindows <= 9; nextNumWindows <= 6; end
            'd3: begin numWindows <= 6; nextNumWindows <= 1; end
            'd4: begin numWindows <= 1; nextNumWindows <= 1; end
            default: begin numWindows <= 0; nextNumWindows <= 0; end
         endcase
    end
    
    logic [RAM_ADDR_WIDTH-1:0] addrRam1, addrRam2, addrRam1_int;
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
    reg [RAM_ADDR_WIDTH-1:0] addrXBegin, addrXEnd, addrYBegin, addrYEnd;
    reg [RAM_ADDR_WIDTH/2-1:0] addrX, addrY;
    wire [RAM_ADDR_WIDTH-1:0] scaledAddr;
    assign scaledAddr = {(addrY*nextNumWindows)/numWindows, (addrX*nextNumWindows)/numWindows};
    
    assign addrRam1_int = weRam1_int?{scaledAddr}:{addrY, addrX};
    assign addrRam2 = weRam2?{scaledAddr}:{addrY, addrX};

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

    logic [7:0] ipguOutQFlipped [79:0];
    
    out_fifo ipgu_out (.clk,.rst_n,    
                        .en((csRam1_int&weRam1_int)|(csRam2_int&weRam2)), //if either RAM is being internally written to, write to buffer as well
                        .d((csRam1_int&weRam1_int)?rdDataRam2:rdDataRam1),
                        .q(ipguOutQFlipped));  
     
/*    generate 
        for(genvar i=0; i<80; i++)
            
    endgenerate*/
    //https://www.amiq.com/consulting/2017/05/29/how-to-pack-data-using-systemverilog-streaming-operators/#reverse_bits
    assign ipguOutBufferQ = {<<{ipguOutQFlipped}}; //potential issue    
   

    //weRam1_int and weRam2
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            weRam1_int <= '0;
            weRam2 <= '0;
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
    typedef enum reg [1:0] {IDLE, RAM1_SRC, RAM2_SRC, WAIT_HEU_RDY} state_t;    
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
            if(windowDone && addrXBegin+20==numWindows[convertI]*20) begin //last pixel of a row of windows be read next
                if(convertDone)
                    addrYBegin <= '0;
                else
                    addrYBegin <= addrYBegin+20;
            end
        end

    //addrXBegin
    always_ff @(posedge clk, negedge rst_n)
        if(!rst_n)
            addrXBegin <= '0;
        else begin
            if(windowDone) begin //last pixel of a windows will be read next
                if(addrXBegin+20==numWindows[convertI]*20)
                    addrXBegin <= '0;
                else
                    addrXBegin <= addrXBegin+20;
            end
        end
    
    //addrXEnd and addrYEnd
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            addrXEnd <= '0;
            addrYEnd <= '0;
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
            else if(windowDone && addrXEnd+1==numWindows[convertI]*20 && addrYEnd+1==numWindows[convertI]*20)
                convertDone <= 1'b1;
        end
    
    end

    assign rdyHeu = convertDone&&convertI==3;//transfer from 60x60 to 20x20 done
    

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
                        nxt_state = WAIT_HEU_RDY;
                    end
                end
                WAIT_HEU_RDY: begin
                    vldIpgu = '1;
                    if(rdyHeu) begin
                        vldIpgu = '0;
                        if(convertDone) begin
                            incConvertI = '1;
                            if(convertI==4)
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
                         nxt_state = WAIT_HEU_RDY;
                     end
                end
                //don't need a default case since all four states are used 
            endcase
        end

endmodule
