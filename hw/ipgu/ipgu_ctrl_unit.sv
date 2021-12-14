module ipgu_ctrl_unit #(RAM_ADDR_WIDTH = 18) 
(
    input                           clk,
    input                           rst_n,
    
    input                           initIpgu,
    input                           rdyHeu,

    input [RAM_ADDR_WIDTH/2-1:0]    addrXBegin, addrXEnd, addrYBegin, addrYEnd,
    input [RAM_ADDR_WIDTH/2-1:0]    addrX, addrY,
 

    output  reg                     vldIpgu,
    output                          rdyIpgu,

    output  reg                     csRam1_int, csRam2_int,
    output  reg                     weRam1_int, weRam2,
    
    output                          windowDone,
    output  reg                     incX,
    output  reg [3:0]               numWindows,
    output  reg [2:0]               convertI   
 
);

    reg convertDone, incConvertI, resetConvertI, clrConvertDone;

    reg csRam1, csRam2;
    reg csRam1_d1, csRam2_d1;

    assign windowDone = addrYEnd==addrY && addrXEnd==addrX+1; //last pixel of a windows will be read next 

    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            rdyIpgu <= '0;
        else if(nxt_state==IDLE||convertDone&&convertI==4)
            rdyIpgu <= '1;
        else if(initIpgu)
            rdyIpgu <= '0;
    end
   
     //assign rdyIpgu = convertDone&&convertI==4;//transfer from 60x60 to 20x20 done
    
    assign csRam1_int = csRam1|csRam2_d1;   
    assign csRam2_int = csRam2|csRam1_d1;


    always_comb begin
        case(convertI) 
            'd0: begin numWindows <= 15; end
            'd1: begin numWindows <= 12; end
            'd2: begin numWindows <= 9;  end
            'd3: begin numWindows <= 6;  end
            'd4: begin numWindows <= 3;  end
            'd5: begin numWindows <= 1;  end
            default: begin numWindows <= 0; end
         endcase
    end

    
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
