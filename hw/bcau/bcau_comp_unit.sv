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
  input [7:0] intensity,

  /*
   * Outputs
   */
  output [7:0] new_intensity
);

reg [14:0] accumulation;
reg [7:0] average_intensity;

wire [14:0] accumulation_d;
wire [7:0] average_intensity_d;

assign average_intensity_d = accumulation / 80;
assign accumulation_d = intensity + accumulation;
assign new_intensity = (intensity > average_intensity) ? ((intensity > 223) ? 255 : intensity + 32) : ((32 > intensity) ? 0 : intensity - 32);

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n || clr_accum)
      accumulation <= 0;
    else if (wr_accum) begin
      accumulation <= accumulation_d;
    end else begin
      accumulation <= accumulation;
    end
end

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
      average_intensity <= 0;
    else if (set_avg) begin
      average_intensity <= average_intensity_d;
    end else begin
      average_intensity <= average_intensity;
    end
end

endmodule