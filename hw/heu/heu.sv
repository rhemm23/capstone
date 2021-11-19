module heu
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input ipgu_out_ready,
    input rdn_in_ready,
    input [7:0] d [4:0][79:0],

    /*
     * Outputs
     */
    output in_ready,
    output out_ready,
    output [7:0] q [4:0][79:0]
  );

  wire [7:0] sum_unit_q [4:0];
  wire [7:0] in_buffer_q [4:0];
  wire [6:0] sums [4:0][255:0];
  
  wire enable_calc;
  wire zero_cnts;

  wire sum_ready;
  wire sum_go;

  wire shift_out;
  wire rotate_in;
  wire write_in;

  heu_ctrl_unit ctrl_unit (
    .clk(clk),
    .rst_n(rst_n),
    .sum_ready(sum_ready),
    .rdn_in_ready(rdn_in_ready),
    .ipgu_out_ready(ipgu_out_ready),
    .sum_go(sum_go),
    .in_ready(in_ready),
    .write_in(write_in),
    .out_ready(out_ready),
    .rotate_in(rotate_in),
    .shift_out(shift_out),
    .zero_cnts(zero_cnts),
    .enable_calc(enable_calc)
  );

  heu_sum_unit sum_unit (
    .clk(clk),
    .rst_n(rst_n),
    .go(sum_go),
    .d(in_buffer_q),
    .sums(sums),
    .ready(sum_ready),
    .q(sum_unit_q)
  );

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
    .d(sum_unit_q),
    .q(q)
  );

  generate
    genvar i;
    for (i = 0; i < 5; i++) begin : calc
      heu_calc_unit calc_unit (
        .clk(clk),
        .rst_n(rst_n),
        .z(zero_cnts),
        .en(enable_calc),
        .d(in_buffer_q[i]),
        .q(sums[i])
      );
    end
  endgenerate
endmodule
