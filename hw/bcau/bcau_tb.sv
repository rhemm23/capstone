module bcau_tb ();

  logic clk;
  logic rst_n;
  logic iru_valid;
  logic heu_ready;
  logic [7:0] iru_results [4:0][79:0];
  logic bcau_valid;
  logic bcau_ready;
  reg [7:0] bcau_results [4:0][79:0];
  reg [7:0] expected_results [4:0][79:0];
  reg [7:0] expected_averages [4:0][4:0];

  bcau dut (
    .clk(clk),
    .rst_n(rst_n),
    .iru_valid(iru_valid),
    .heu_ready(heu_ready),
    .iru_results(iru_results),
    .bcau_valid(bcau_valid),
    .bcau_ready(bcau_ready),
    .bcau_results(bcau_results)
  );

  integer i,j,k, accum[4:0];
  


  initial begin
    clk = 0;
    rst_n = 0;
    iru_valid = 0;
    heu_ready = 0;
    for (j = 0; j < 5; j = j + 1) begin
      for (k = 0; k < 80; k = k + 1) begin
        iru_results[j][k] = 0;
        expected_results[j][k] = 0;
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

      begin 
        for (i = 0; i < 10; i = i + 1) begin

          // Find Averages and generate input
          for (j = 0; j < 5; j = j + 1) begin
            for (k = 0; k < 5; k = k + 1) accum[k] = 0;
            for (k = 0; k < 80; k = k + 1) begin
              iru_results[j][k] = $random;
              accum[((k % 20) / 4)] += iru_results[j][k];
            end
            for (k = 0; k < 5; k = k + 1) expected_averages[j][k] = (accum[k] / 16);
          end

          // Calculating expected results
          for (j = 0; j < 5; j = j + 1) begin
            for (k = 0; k < 80; k = k + 1) begin

              if (iru_results[j][k] > expected_averages[j][((k % 20) / 4)]) begin
                if (iru_results[j][k] > (255-32)) begin
                  expected_results[j][k] = 255;
                end else begin
                  expected_results[j][k] = iru_results[j][k] + 32;
                end
              end else begin
                if (iru_results[j][k] < 32) begin
                  expected_results[j][k] = 0;
                end else begin
                  expected_results[j][k] = iru_results[j][k] - 32;
                end
              end

            end
          end

          // Checking results
          iru_valid = 1;
          @(posedge clk);
          iru_valid = 0;
          @(posedge bcau_valid);
          for (j = 0; j < 5; j = j + 1) begin
            for (k = 0; k < 80; k = k + 1) begin
              if (expected_results[j][k] !== bcau_results[j][k]) begin
                $display("Error: Invalid output value for bcau_results[%d][%d]: %h, expected %h", j, k, bcau_results[j][k], expected_results[j][k]);
                $stop();
              end
            end
          end
          @(posedge clk);
          heu_ready = 1;
          @(posedge bcau_ready);
          @(posedge clk);
          heu_ready = 0;
        end
        disable timeout1;
      end
    join
    $display("BCAU tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule