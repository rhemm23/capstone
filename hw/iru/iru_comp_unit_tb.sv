module iru_comp_unit_tb();

  logic clk;
  logic rst_n;

  logic [35:0] rnn_out;

  logic [4:0] row_d;
  logic [4:0] col_d;

  wire [4:0] row_q;
  wire [4:0] col_q;

  wire signed [8:0] cos_q;
  wire signed [8:0] sin_q;

  wire valid;

  int next_x;
  int next_y;

  logic pred_valid;

  iru_comp_unit comp_unit (
    .rnn_out(rnn_out),
    .row_d(row_d),
    .col_d(col_d),
    .valid(valid),
    .row_q(row_q),
    .col_q(col_q)
  );

  iru_cos_lut cos_lut (
    .d(rnn_out),
    .q(cos_q)
  );

  iru_sin_lut sin_lut (
    .d(rnn_out),
    .q(sin_q)
  );

  initial begin
    clk = 0;
    rst_n = 1;
    for (int i = 0; i < 36; i++) begin

      rst_n = 0;
      #1;
      rst_n = 1;
      #1;

      rnn_out = '0;
      rnn_out[i] = 1'b1;
      #5;
      for (int y = 0; y < 20; y++) begin
        for (int x = 0; x < 20; x++) begin
          next_x = ((x * cos_q) - (y * sin_q)) >>> 7;
          next_y = ((x * sin_q) + (y * cos_q)) >>> 7;
          pred_valid = (next_x >= 0) &&
                       (next_y >= 0) &&
                       (next_x < 20) &&
                       (next_y < 20);
          col_d = x;
          row_d = y;
          #5;
          if (pred_valid !== valid) begin
            $display("Test failed: vld = %b, col = %d, row = %d Expected vld = %b, col = %d, row = %d", valid, col_q, row_q, pred_valid, next_x, next_y);
            $stop();
          end else if (valid) begin
            if (col_q !== next_x || row_q !== next_y) begin
              $display("Test failed: vld = %b, col = %d, row = %d Expected vld = %b, col = %d, row = %d", valid, col_q, row_q, pred_valid, next_x, next_y);
              $stop();
            end
          end
        end
      end
    end
    $display("Tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
