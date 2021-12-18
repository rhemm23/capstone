module heu_tb ();

  logic [7:0] d [4:0][79:0];

  logic ipgu_out_ready;
  logic rdn_in_ready;
  logic rst_n;
  logic clk;

  wire [7:0] q [4:0][79:0];
  wire out_ready;
  wire in_ready;

  int sums [255:0];
  int cdist [255:0];
  int results [4:0][79:0];

  heu dut (
    .clk(clk),
    .rst_n(rst_n),
    .ipgu_out_ready(ipgu_out_ready),
    .rdn_in_ready(rdn_in_ready),
    .d(d),
    .in_ready(in_ready),
    .out_ready(out_ready),
    .q(q)
  );

  initial begin
  
    clk = 0;
    rst_n = 1;

    ipgu_out_ready = 0;
    rdn_in_ready = 0;

    // Run 50 random tests
    for (int i = 0; i < 50; i++) begin

      // Reset
      #5;
      rst_n = 0;
      #5;
      rst_n = 1;
      #5;

      // Setup random input
      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          d[k][j] = $urandom();
        end
      end

      sums = '{ 256 { 0 } };
      cdist = '{ 256 { 0 } };

      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          results[k][j] = 0;
        end
      end
      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          sums[d[k][j]]++;
        end
      end
      cdist[0] = sums[0];
      for (int j = 1; j < 256; j++) begin
        cdist[j] = cdist[j - 1] + sums[j];
      end
      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          results[k][j] = (cdist[d[k][j]] * 255) / 400;
        end
      end

      ipgu_out_ready = 1;
      @(posedge clk);
      
      ipgu_out_ready = 0;
      @(posedge out_ready);

      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          if (results[k][j] != q[k][j]) begin
            $display("Error: Invalid output value for HEU %h, expected %h", q[k][j], d[k][j]);
            $stop();
          end
        end
      end
    end

    $display("HEU tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
