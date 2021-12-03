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

  for (int i = 0; i < 50; i++) begin
    for (int j = 0; j < 36; j++) begin

      rst_n = 1;
      #1;
      rst_n = 0;
      #1;
      rst_n = 1;
      #1;

      for (int k = 0; k < 36; k++) begin
        rnn_out[k] = (j == k) ? 1'b1 : 1'b0;
      end
      for (int k = 0; k < 5; k++) begin
        for (int l = 0; l < 80; l++) begin
          d[k][l] = $urandom();
          byte cur_x = (k * 4) + (l / 20);
          byte cur_y = (l % 20);
          byte new_x = 
        end
      end

      int rotated [4:0][79:0];
      for (int k = 0; k < 5; k++) begin
        for (int l = 0; l < 80; l++) begin
          pixels[k][l] = $urandom();
        end
      end
    end
  end

  always
    #5 clk = ~clk;

endmodule
