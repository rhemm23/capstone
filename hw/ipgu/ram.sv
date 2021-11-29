//https://www.chipverify.com/verilog/verilog-single-port-ram

module single_port_sync_ram;
# (     parameter ADDR_WIDTH_X  = $clog2(DEPTH_X),
        parameter ADDR_WIDTH_Y  = $clog2(DEPTH_Y), //X and Y
        parameter DATA_WIDTH    = 8,
        parameter DEPTH_X       = 300,
        parameter DEPTH_Y       = 300
  )
( input                                     clk,
  input [(ADDR_WIDTH_X+ADDR_WIDTH_Y)-1:0]   addr,           //={addrY,addrX}
  inout [DATA_WIDTH-1:0]                    rdData, wrData,
  input                                     cs,
  input                                     we,
  );

wire [ADDR_WIDTH_X-1:0] addr_x;
wire [ADDR_WIDTH_Y-1:0] addr_y;

assign addr_x = addr[ADDR_WIDTH_X-1:0];
assign addr_y = addr[ADDR_WIDTH_X+:ADDR_WIDTH_Y];

reg [DATA_WITDH-1:0] mem [DEPTH_X][DEPTH_Y];

always @ (posedge clk) begin
    if(cs&we)
        mem[addr_x][addr_y] <= wrData;
    if(cs&!we)
        rdData <= mem[addr_x][addr_y];
end

endmodule
