module bcau_comp_unit
(
  /*
   * Inputs
   */
  input clk,
  input rst_n,
  input wr_accum,
  input set_avg,
  input clr_accum,
  input [2:0] block_sel,
  input [7:0] intensity,

  /*
   * Outputs
   */
  output [7:0] new_intensity
);

reg [11:0] accumulation [4:0];
reg [7:0] average_intensity [4:0];

wire [11:0] accumulation_d [4:0];
wire [7:0] average_intensity_d [4:0];


assign average_intensity_d[0] = accumulation[0] >> 4;
assign average_intensity_d[1] = accumulation[1] >> 4;
assign average_intensity_d[2] = accumulation[2] >> 4;
assign average_intensity_d[3] = accumulation[3] >> 4;
assign average_intensity_d[4] = accumulation[4] >> 4;

assign accumulation_d[0] = intensity + accumulation[0];
assign accumulation_d[1] = intensity + accumulation[1];
assign accumulation_d[2] = intensity + accumulation[2];
assign accumulation_d[3] = intensity + accumulation[3];
assign accumulation_d[4] = intensity + accumulation[4];


assign new_intensity = (intensity > average_intensity[block_sel]) ? ((intensity > 223) ? 255 : intensity + 32) : ((32 > intensity) ? 0 : intensity - 32);

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n || clr_accum)
      for (int k = 0; k < 5; k = k + 1) accumulation[k] <= 0;
    else if (wr_accum) begin
      accumulation[block_sel] <= accumulation_d[block_sel];
    end 
end

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
      for (int k = 0; k < 5; k = k + 1) average_intensity[k] <= 0;
    else if (set_avg) begin
      for (int k = 0; k < 5; k = k + 1) average_intensity[k] <= average_intensity_d[k];
    end
end

endmodule