module iru_tb();

  int expected [4:0][79:0];

  int buffer;
  int index;

  int cur_x;
  int cur_y;

  int new_x;
  int new_y;

  logic [7:0] d [4:0][79:0];

  logic [35:0] rnn_out;

  logic [4:0] row_d;
  logic [4:0] col_d;

  logic rnn_out_ready;
  logic bcau_in_ready;

  logic rst_n;
  logic clk;

  wire [7:0] q [4:0][79:0];

  wire [4:0] row_q;
  wire [4:0] col_q;

  wire out_ready;
  wire in_ready;
  wire valid;

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

  iru_comp_unit comp_unit (
    .rnn_out(rnn_out),
    .row_d(row_d),
    .col_d(col_d),
    .valid(valid),
    .row_q(row_q),
    .col_q(col_q)
  );

  initial begin

    clk = 0;
    rst_n = 1;

    rnn_out_ready = 0;
    bcau_in_ready = 0;

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
            expected[k][l] = 0;
          end
        end
        for (int k = 0; k < 5; k++) begin
          for (int l = 0; l < 80; l++) begin
            d[k][l] = $urandom();
            cur_x = (k * 4) + (l / 20);
            cur_y = (l % 20);
            row_d = cur_y[4:0];
            col_d = cur_x[4:0];
            #5;
            if (valid) begin
              expected[row_q][col_q] = d[k][l];
            end
          end
        end

        @(posedge clk);
        rnn_out_ready = 1;
        @(posedge clk);
        rnn_out_ready = 0;
        @(posedge clk);

        @(posedge out_ready);
        for (int k = 0; k < 5; k++) begin
          for (int l = 0; l < 80; l++) begin
            if (q[k][l] !== expected[k][l]) begin
              $write("Q\n\n");
              for (int x = 0; x < 5; x++) begin
                for (int y = 0; y < 80; y++) begin
                  if (y > 0) begin
                    $write(" ");
                  end
                  $write("%h", q[x][y]);
                  if ((y + 1) % 20 == 0) begin
                    $write("\n");
                  end
                end
                $write("\n");
              end
              $write("EXPECTED\n\n");
              for (int x = 0; x < 5; x++) begin
                for (int y = 0; y < 80; y++) begin
                  if (y > 0) begin
                    $write(" ");
                  end
                  $write("%h", expected[x][y]);
                end
                $write("\n");
              end
              $display("Error: Expected %h, actual %h", expected[k][l], q[k][l]);
              $stop();
            end
          end
        end
      end
    end

    $display("IRU tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
