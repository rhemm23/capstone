module bcau
(
  /*
    * Inputs
    */
  input clk,
  input rst_n,
  input iru_valid,
  input heu_ready,
  input [7:0] iru_results [4:0][79:0],

  /*
    * Outputs
    */
  output bcau_valid,
  output bcau_ready,
  output [7:0] bcau_results [4:0][79:0]
);
  /*
   * ~163 cycles till bcau_valid is asserted
   */

wire wr_in_all, cir_shft, shft_out, clr_accum;
wire [7:0] new_intensity [4:0];
wire [7:0] in_buffer_q [4:0];

bcau_ctrl_unit ctrl_unit (
  .clk(clk),
  .rst_n(rst_n),
  .iru_valid(iru_valid),
  .heu_ready(heu_ready),
  .bcau_valid(bcau_valid),
  .bcau_ready(bcau_ready),
  .wr_in_all(wr_in_all),
  .cir_fifo(cir_shft),
  .wr_accum(wr_accum),
  .set_avg(set_avg),
  .shft_out(shft_out),
  .clr_accum(clr_accum)
);
generate
  genvar i;
  for (i = 0; i < 5; i++) begin : comp_units
    bcau_comp_unit comp_unit (
      .clk(clk),
      .rst_n(rst_n),
      .wr_accum(wr_accum),
      .set_avg(set_avg),
      .intensity(in_buffer_q[i]),
      .clr_accum(clr_accum),
      .new_intensity(new_intensity[i])
    );
  end
endgenerate
in_buffer in_buf (
  .clk(clk),
  .rst_n(rst_n),
  .wr(wr_in_all),
  .en(cir_shft),
  .d(iru_results),
  .q(in_buffer_q)
);
out_buffer out_buf (
  .clk(clk),
  .rst_n(rst_n),
  .en(shft_out),
  .d(new_intensity),
  .q(bcau_results)
);

endmodule