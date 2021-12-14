module rdn_fp_tb ();
// Done

  logic clk;
  logic rst_n;
  logic heu_out_valid;
  logic iru_in_ready;
  logic [7:0] d [4:0][79:0];
  logic load_weights;
  logic mem_ready;
  logic [63:0] mem_data [7:0];

  logic [35:0] expected_angle;
  logic [7:0] expected_q [4:0][79:0];

  rdn_fp dut (
    .clk(clk),
    .rst_n(rst_n),
    .load_weights(load_weights),
    .mem_ready(mem_ready),
    .heu_out_valid(heu_out_valid),
    .iru_in_ready(iru_in_ready),
    .d(d),
    .mem_data(mem_data),

    .in_ready(in_ready),
    .out_ready(out_ready),
    .weight_valid(weight_valid),
    .req_mem(req_mem),
    .angle_out(angle_out),
    .q(q)
  );

  integer i, j, k, accum;

  initial begin
    clk = 0;
    rst_n = 0;
    heu_out_valid = 0;
    iru_in_ready = 0;
    expected_angle = 0;
    for (j = 0; j < 5; j = j + 1) begin
      for (k = 0; k < 80; k = k + 1) begin
        d[j][k] = 0;
        expected_q[j][k] = 0;
      end
    end 

    // reset
    @(posedge clk);
    rst_n = 1;

    fork
      begin : timeout1
        repeat(10000) @(posedge clk);
        $display("ERROR: timed out");
        $stop();
      end

    join
    $display("BCAU tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule