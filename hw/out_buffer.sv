module out_buffer
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input en,
    input [7:0] d [4:0],

    /*
     * Outputs
     */
    output [7:0] q [4:0][79:0]
  );

  out_fifo fifos [4:0] (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .d(d),
    .q(q)
  );

endmodule
