module iru_comp_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input [35:0] rnn_out,
    input [4:0] row_d,
    input [4:0] col_d,

    /*
     * Outputs
     */
    output valid,
    output [4:0] row_q,
    output [4:0] col_q
  );

  wire signed [15:0] x_exp;
  wire signed [15:0] y_exp;

  wire signed [8:0] cos_q;
  wire signed [8:0] sin_q;

  iru_cos_lut cos_lut (
    .d(rnn_out),
    .q(cos_q)
  );

  iru_sin_lut sin_lut (
    .d(rnn_out),
    .q(sin_q)
  );

  assign x_exp = ($signed({ 1'b0, col_d }) * cos_q) - ($signed({ 1'b0, row_d }) * sin_q);
  assign y_exp = ($signed({ 1'b0, col_d }) * sin_q) + ($signed({ 1'b0, row_d }) * cos_q);

  assign col_q = (x_exp[15] ? -x_exp : x_exp)[11:7];
  assign row_q = (y_exp[15] ? -y_exp : y_exp)[11:7];

  assign valid = ~x_exp[15] &&
                 ~y_exp[15] &&
                 (col_q < 20) &&
                 (row_q < 20);

endmodule
