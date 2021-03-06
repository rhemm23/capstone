//https://www.chipverify.com/verilog/verilog-single-port-ram

module ram
# (     parameter ADDR_WIDTH_X  = 9,
        parameter ADDR_WIDTH_Y  = 9, //X and Y
        parameter DATA_WIDTH    = 8,
        parameter DEPTH_X       = 300,
        parameter DEPTH_Y       = 300
  )
( input                                     clk,
  input [(ADDR_WIDTH_X+ADDR_WIDTH_Y)-1:0]   addr,           //={addrY,addrX}
  input [DATA_WIDTH-1:0]                    wrData,
  output reg [DATA_WIDTH-1:0]               rdData,
  input                                     cs,
  input                                     we
  );

wire [ADDR_WIDTH_X-1:0] addr_x;
wire [ADDR_WIDTH_Y-1:0] addr_y;

assign addr_x = addr[ADDR_WIDTH_X-1:0];
assign addr_y = addr[ADDR_WIDTH_X+:ADDR_WIDTH_Y];

reg [DATA_WIDTH-1:0] mem [DEPTH_Y-1:0][DEPTH_X-1:0];

always @ (posedge clk) begin
    if(cs&we)
        mem[addr_y][addr_x] <= wrData;
    if(cs&!we)
        rdData <= mem[addr_y][addr_x];
end

endmodule
