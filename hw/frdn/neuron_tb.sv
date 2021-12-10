module neuron_tb();

  logic clk;
  logic rst_n;
  logic start;
  logic write_weight;
  logic [8:0] weight_sel;
  logic [63:0] weight_bus;
  logic [63:0] data [399:0];

  wire done;
  wire [63:0] activation;

  real bias;
  real accum;
  real weights [399:0];

  logic [7:0] temp;

  neuron dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data(data),
    .write_weight(write_weight),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activation(activation),
    .done(done)
  );

  initial begin
    
    clk = 0;
    rst_n = 1;

    start = 0;
    write_weight = 0;

    #5;
    rst_n = 0;
    #5;
    rst_n = 1;
    #5;

    temp = $urandom();
    bias = $itor(temp) / 512.0;
    for (integer i = 0; i < 400; i++) begin
      temp = $urandom();
      data[i] = $realtobits($itor(temp) / 256.0);

      temp = $urandom();
      weights[i] = $itor(temp) / 512000.0;
    end

    for (integer i = 0; i < 400; i++) begin
      write_weight = 1;
      weight_sel = i + 1;
      weight_bus = $realtobits(weights[i]);
      @(posedge clk);
    end

    write_weight = 1;
    weight_sel = 0;
    weight_bus = $realtobits(bias);
    @(posedge clk);

    write_weight = 0;

    @(posedge clk);
    @(posedge clk);

    accum = bias;
    for (integer i = 0; i < 400; i++) begin
      accum += $bitstoreal(data[i]) * weights[i];
    end
    accum = $tanh(accum);

    $display("Calculated: %e", accum);

    start = 1;
    @(posedge clk);
    start = 0;
    @(posedge done);

    $display("Result: %e", $bitstoreal(activation));

    $stop();

  end

  always
    #5 clk = ~clk;

endmodule
