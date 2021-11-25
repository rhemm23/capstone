module b_neuron
  #(
    INPUTS = 15
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input z,
    input en,
    input wr_weights,
    input signed [8:0] d,
    input signed [8:0] bias_d,
    input signed [8:0] weights_d [INPUTS-1:0],

    /*
     * Outputs
     */
    output signed [8:0] q
  );

  wire [31:0] accum_abs;
  wire [7:0] accum_rnd;

  reg signed [8:0] weights [INPUTS-1:0];
  reg signed [8:0] bias;

  reg signed [31:0] accum;

  reg [4:0] cnt;

  tanh_lut lut (
    .d(accum_rnd),
    .q(q[7:0])
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weights <= '{ INPUTS { 0 } };
      accum <= 0;
      bias <= 0;
      cnt <= 0;
    end else if (wr_weights) begin
      weights <= weights_d;
      bias <= bias_d;
    end else if (z) begin
      accum <= bias <<< 2;
      cnt <= 0;
    end else if (en) begin
      if (cnt < INPUTS) begin
        accum <= accum + (d * weights[cnt]);
        cnt <= cnt + 1;
      end
    end
  end

  assign accum_abs = accum[31] ? -accum : accum;

  assign accum_rnd = &accum_abs[30:10] ? 8'hFF : accum_abs[9:2];

  assign q[8] = accum[31];

endmodule
