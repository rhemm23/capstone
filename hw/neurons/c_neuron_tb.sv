module c_neuron_tb();

  logic clk;
  logic rst_n;

  logic z;
  logic en;
  logic wr_weights;

  logic signed [8:0] bias;
  logic signed [8:0] weights [14:0];

  logic signed [8:0] d;
  logic signed [8:0] inputs [14:0];

  wire q;

  logic signed [31:0] sum;
  logic signed [31:0] sum_abs;

  logic [7:0] sum_rnd;

  c_neuron dut (
    .clk(clk),
    .rst_n(rst_n),
    .z(z),
    .en(en),
    .wr_weights(wr_weights),
    .d(d),
    .bias_d(bias),
    .weights_d(weights),
    .q(q)
  );

  initial begin

    clk = 0;
    rst_n = 1;

    z = 0;
    en = 0;
    wr_weights = 0;

    // Run 50 random tests
    for (int i = 0; i < 50; i++) begin

      // Reset
      #5;
      rst_n = 0;
      #5;
      rst_n = 1;
      #5;

      // Random biases, weights, and inputs
      bias = $random;
      for (int j = 0; j < 15; j++) begin
        weights[j] = $random;
        inputs[j] = $random;
      end


      // Write weights
      @(posedge clk);
      wr_weights = 1;
      @(posedge clk);

      wr_weights = 0;
      @(posedge clk);

      // Zero out sum and cnt
      z = 1;
      @(posedge clk);

      z = 0;
      @(posedge clk);

      // Start
      en = 1;
      for (int j = 0; j < 15; j++) begin
        d = inputs[j];
        @(posedge clk);
      end
      en = 0;

      // Calc expected
      sum = bias <<< 2;
      for (int j = 0; j < 15; j++) begin
        sum += inputs[j] * weights[j];
      end

      sum_abs = (sum < 0) ? -sum : sum;
      sum_rnd = (|sum_abs[30:10]) ? 8'hFF : sum_abs[9:2];

      // Let propagate
      #1;

      // Check
      if (q !== sum_rnd[7]) begin
        $display("Error: Invalid output value for c neuron %d, expected %d", q, sum_rnd[7]);
        $stop();
      end
    end

    $display("C neuron tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
