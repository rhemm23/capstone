module iru_visual_tb();

  logic [7:0] d [19:0][19:0];
  logic [7:0] q [19:0][19:0];

  logic [35:0] rnn_out;

  logic [4:0] row_d;
  logic [4:0] col_d;

  wire [4:0] row_q;
  wire [4:0] col_q;

  wire valid;

  int in_fd;
  int out_fd;

  iru_comp_unit comp_unit (
    .rnn_out(rnn_out),
    .row_d(row_d),
    .col_d(col_d),
    .valid(valid),
    .row_q(row_q),
    .col_q(col_q)
  );

  initial begin

    rnn_out = 36'b000000000000000000000000000000000010;

    in_fd = $fopen("image.bin", "rb");

    if (in_fd == 0) begin
      $display("Could not open image file");
      $stop();
    end
    if ($fread(d, in_fd) != 400) begin
      $display("Could not read from image file");
      $stop();
    end

    for (int y = 0; y < 20; y++) begin
      for (int x = 0; x < 20; x++) begin
        q[y][x] = '0;
      end
    end
    for (int y = 0; y < 20; y++) begin
      for (int x = 0; x < 20; x++) begin
        row_d = y;
        col_d = x;
        #5;
        if (valid) begin
          q[row_q][col_q] = d[y][x];
        end
      end
    end

    out_fd = $fopen("result.bin", "wb+");

    if (out_fd == 0) begin
      $display("Could not open result file");
      $stop();
    end
    for (int y = 0; y < 20; y++) begin
      for (int x = 0; x < 20; x++) begin
        $fwrite(out_fd, "%c", q[y][x]);
      end
    end

    $fclose(in_fd);
    $fclose(out_fd);

    $display("Done");
    $stop();
  end

endmodule
