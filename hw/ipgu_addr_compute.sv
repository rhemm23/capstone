module ipgu_addr_compute #(RAM_ADDR_WIDTH = 18) 
(
    input clk,
    input rst_n,
    input incX,
    input windowDone,
    input [3:0] numWindows,
    input [2:0] convertI,
    output reg [RAM_ADDR_WIDTH/2-1:0] addrXBegin, addrXEnd, addrYBegin, addrYEnd,
    output reg [RAM_ADDR_WIDTH/2-1:0] addrX, addrY,
    output reg [$clog2(240)-1:0] scaledX, scaledY

);

    wire [$clog2(240)-1:0] lutAddrX, lutAddrY;
 
    ipgu_mult_lut lutX(.scaleNum(convertI), .addr(addrX), .addrOut(lutAddrX));
    ipgu_mult_lut lutY(.scaleNum(convertI), .addr(addrY), .addrOut(lutAddrY));

    //scaledAddr lut output transferred to scaled Addr with 1 cycle delay
    always_ff @(posedge clk) begin
        scaledX <= lutAddrX;
        scaledY <= lutAddrY;
    end

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
                    addrYBegin <= addrYBegin + 10;
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
            addrXEnd <= addrXBegin+19;
            addrYEnd <= addrYBegin+19;
        end
    end
 

endmodule
