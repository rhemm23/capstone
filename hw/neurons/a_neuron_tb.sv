module a_neuron_tb();

  logic clk;
  logic rst_n;

  logic z;
  logic en;
  logic wr_weights;

  logic signed [8:0] bias;
  logic signed [8:0] weights [399:0];

  logic [7:0] d [4:0];
  logic [7:0] inputs [399:0];

  wire signed [8:0] q;

  logic signed [31:0] sum;
  logic signed [31:0] sum_abs;

  logic [7:0] tan_in;
  wire [7:0] tan_out;

  tanh_lut lut (
    .d(tan_in),
    .q(tan_out)
  );

  a_neuron dut (
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
    tan_in = 0;
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
      for (int j = 0; j < 400; j++) begin
        weights[j] = $random;
        inputs[j] = $urandom;
      end

      // Write weights
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
      for (int j = 0; j < 80; j++) begin
        for (int k = 0; k < 5; k++) begin
          d[k] = inputs[(j * 5) + k];
        end
        @(posedge clk);
      end
      en = 0;

      // Calc expected
      sum = bias;
      for (int j = 0; j < 400; j++) begin
        sum += $signed({ 1'b0, inputs[j] }) * weights[j];
      end

      sum_abs = (sum < 0) ? -sum : sum;
      tan_in = &sum_abs[30:10] ? 8'hFF : sum_abs[9:2];

      // Let tanh lut propagate
      #1;

      // Check
      if (q !== { sum[31], tan_out }) begin
        $display("Error: Invalid output value for a neuron %h, expected %h", q, { sum[31], tan_out });
        $stop();
      end
    end

    $display("A neuron tests passed!");
    $stop();
  end

  always
    #5 clk = ~clk;

endmodule
