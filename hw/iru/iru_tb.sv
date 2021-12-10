module iru_tb();

  logic [7:0] d [4:0][79:0];

  logic [35:0] rnn_out;

  logic rnn_out_ready;
  logic bcau_in_ready;

  logic rst_n;
  logic clk;

  wire [7:0] q [4:0][79:0];

  wire out_ready;
  wire in_ready;

  int in_fd;
  int out_fd;

  iru dut (
    .clk(clk),
    .rst_n(rst_n),
    .rnn_out_ready(rnn_out_ready),
    .bcau_in_ready(bcau_in_ready),
    .rnn_out(rnn_out),
    .d(d),
    .in_ready(in_ready),
    .out_ready(out_ready),
    .q(q)
  );

  initial begin

    clk = 0;
    rst_n = 1;

    rnn_out_ready = 0;
    bcau_in_ready = 0;

    rnn_out = 36'b000000000000000001000000000000000000;

    in_fd = $fopen("image.bin", "rb");

    if (in_fd == 0) begin
      $display("Could not open image file");
      $stop();
    end
    if ($fread(d, in_fd) != 400) begin
      $display("Could not read from image file");
      $stop();
    end

    rst_n = 0;
    #1;
    rst_n = 1;
    #1;

    @(posedge clk);
    rnn_out_ready = 1;
    @(posedge clk);
    rnn_out_ready = 0;
    @(posedge clk);

    @(posedge out_ready);

    out_fd = $fopen("result.bin", "wb+");

    if (out_fd == 0) begin
      $display("Could not open result file");
      $stop();
    end
    for (int y = 0; y < 5; y++) begin
      for (int x = 0; x < 80; x++) begin
        $fwrite(out_fd, "%c", q[y][x]);
      end
    end

    $fclose(in_fd);
    $fclose(out_fd);

    $display("Done");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
