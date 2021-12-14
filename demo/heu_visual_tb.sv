module heu_visual_tb();

  logic [7:0] d [4:0][79:0];

  logic ipgu_out_ready;
  logic rdn_in_ready;
  logic rst_n;
  logic clk;

  wire [7:0] q [4:0][79:0];
  wire out_ready;
  wire in_ready;

  int in_fd;
  int out_fd;

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

    in_fd = $fopen("image.bin", "rb");

    if (in_fd == 0) begin
      $display("Could not open image file");
      $stop();
    end
    for (int i = 0; i < 5; i++) begin
      if ($fread(in_fd, d[i], 0, 80) != 80) begin
        $display("Could not read from image file");
        $stop();
      end
    end

    // Reset
    #5;
    rst_n = 0;
    #5;
    rst_n = 1;
    #5;

    ipgu_out_ready = 1;
    @(posedge clk);
    
    ipgu_out_ready = 0;
    @(posedge out_ready);

    out_fd = $fopen("heu_result.bin", "wb+");

    if (out_fd == 0) begin
      $display("Could not open result file");
      $stop();
    end
    for (int i = 0; i < 5; i++) begin
      for (int j = 0; j < 80; j++) begin
        $fwrite(out_fd, "%c", q[i][j]);
      end
    end

    $fclose(in_fd);
    $fclose(out_fd);
  end

  always
    #5 clk = ~clk;

endmodule
