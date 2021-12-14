module iru_in_buffer
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input wr,
    input en,
    input [7:0] d [4:0][79:0],

    /*
     * Outputs
     */
    output [7:0] q [4:0]
  );

  rev_fifo fifos [4:0] (
    .clk(clk),
    .rst_n(rst_n),
    .wr(wr),
    .en(en),
    .d(d),
    .q(q)
  );

endmodule
