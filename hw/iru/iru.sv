module iru
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input rnn_out_ready,
    input bcau_in_ready,
    input [35:0] rnn_out,
    input [7:0] d [4:0][79:0],

    /*
     * Outputs
     */
    output in_ready,
    output out_ready,
    output [7:0] q [4:0][79:0]
  );

  wire [7:0] in_buffer_q [4:0];

  wire [4:0] row_d [4:0];
  wire [4:0] col_d [4:0];

  wire [4:0] row_q [4:0];
  wire [4:0] col_q [4:0];

  wire valid [4:0];
  wire wr [4:0];

  wire write_out;
  wire zero_out;

  wire rotate_in;
  wire write_in;

  iru_comp_unit comp_units [4:0] (
    .rnn_out(rnn_out),
    .row_d(row_d),
    .col_d(col_d),
    .valid(valid),
    .row_q(row_q),
    .col_q(col_q)
  );

  iru_ctrl_unit ctrl_unit (
    .clk(clk),
    .rst_n(rst_n),
    .rnn_out_ready(rnn_out_ready),
    .bcau_in_ready(bcau_in_ready),
    .row_sel(row_d),
    .col_sel(col_d),
    .in_ready(in_ready),
    .out_ready(out_ready),
    .write_in(write_in),
    .rotate_in(rotate_in),
    .zero_out(zero_out),
    .write_out(write_out)
  );

  iru_in_buffer in_buffer (
    .clk(clk),
    .rst_n(rst_n),
    .wr(write_in),
    .en(rotate_in),
    .d(d),
    .q(in_buffer_q)
  );

  iru_out_buffer out_buf (
    .clk(clk),
    .rst_n(rst_n),
    .z(zero_out),
    .wr(wr),
    .d(in_buffer_q),
    .row(row_q),
    .col(col_q),
    .q(q)
  );

  generate
    genvar i;
    for (i = 0; i < 5; i++) begin
      assign wr[i] = valid[i] & write_out;
    end
  endgenerate

endmodule
