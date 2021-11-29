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
    output  vldIpgu,
);

    wire [RAM_ADDR_WIDTH-1:0] addrRam1;
    assign addrRam1 = csRam1_ext?addrRam1_ext:addrRam1_int;

    wire [RAM_DATA_WIDTH-1:0] wrDataRam1;
    assign wrDataRam1 = csRam1_ext?wrDataRam1_ext:wrDataRam1_int;

    wire weRam1;
    assign weRam1 = csRam1_ext?weRam1_ext:weRam1_int;

    ram #(.DATA_WIDTH(RAM_DATA_WIDTH)) ram1 (
        .clk, .addr(addrRam1), .rdData(), .wrData(wrDataRam1), .
        .cs(csRam1_ext|csRam1), .we(weRam1)
    );
    
    ram #(.DEPTH_X(240), .DEPTH_Y(240), .DATA_WIDTH(RAM_DATA_WIDTH)) ram2 (
        .clk, .addr(addrRam1), .rdData(), .wrData(wrDataRam1), .
        .cs(csRam1_ext|csRam1), .we(weRam1)
    );
	///////////////////////////////////////////
    // SM states //
	///////////////////////////////////////////
	typedef enum reg [1:0] {IDLE, RAM1_SRC, RAM2_SRC, DONE} state_t;	
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
            weRam1_int = 1'b0;
            incConvertI = '0;            	
			case (state)
				IDLE : begin
                    if(initIpgu)
                        nxt_state = RAM1_SRC;
				end 
                RAM1_SRC: begin
                    csRam1 = '1;
                    weRam2 = '1;
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
                    end
                end
                RAM2_SRC: begin
                    
                end
				//don't need a default case since all four states are used 
			endcase
		end

endmodule
