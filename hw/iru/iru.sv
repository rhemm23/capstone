module iru
  (
    input clk,
    input rst_n,
    input rnn_out_ready,
    input bcau_in_ready,
    input [35:0] rnn_res,
    input [7:0] d [4:0][79:0],

    output [7:0] q [4:0][79:0]
  );

  wire [7:0] in_buffer_q [4:0];

  wire rotate_in;
  wire write_in;
  wire shift_out;

  in_buffer in_buf (
    .clk(clk),
    .rst_n(rst_n),
    .wr(write_in),
    .en(rotate_in),
    .d(d),
    .q(in_buffer_q)
  );

  out_buffer out_buf (
    .clk(clk),
    .rst_n(rst_n),
    .en(shift_out),
    .d(),
    .q(q)
  );

endmodule
