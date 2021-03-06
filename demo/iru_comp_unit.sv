module iru_comp_unit
  (
    /*
     * Inputs
     */
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

  wire signed [37:0] row_m_cos;
  wire signed [37:0] row_m_sin;

  wire signed [37:0] col_m_cos;
  wire signed [37:0] col_m_sin;

  wire signed [15:0] calc_row;
  wire signed [15:0] calc_col;

  wire signed [31:0] cos_q;
  wire signed [31:0] sin_q;

  wire signed [5:0] row_shift;
  wire signed [5:0] col_shift;

  iru_cos_lut cos_lut (
    .d(rnn_out),
    .q(cos_q)
  );

  iru_sin_lut sin_lut (
    .d(rnn_out),
    .q(sin_q)
  );

  assign row_shift = $signed({ 1'b0, row_d }) - 10;
  assign col_shift = $signed({ 1'b0, col_d }) - 10;

  assign row_m_cos = (row_shift * cos_q) >>> 30;
  assign row_m_sin = (row_shift * sin_q) >>> 30;

  assign col_m_cos = (col_shift * cos_q) >>> 30;
  assign col_m_sin = (col_shift * sin_q) >>> 30;

  assign calc_col = (col_m_cos - row_m_sin) + 10;
  assign calc_row = (col_m_sin + row_m_cos) + 10;

  assign col_q = calc_col[4:0];
  assign row_q = calc_row[4:0];

  assign valid = (calc_col >= 0) &&
                 (calc_row >= 0) &&
                 (calc_col < 20) &&
                 (calc_row < 20);

endmodule
