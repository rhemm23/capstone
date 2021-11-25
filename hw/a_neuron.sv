module a_neuron
  #(
    INPUTS = 400
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
    input [7:0] d [4:0],
    input signed [8:0] bias_d,
    input signed [8:0] weights_d [INPUTS-1:0],

    /*
     * Outputs
     */
    output signed [8:0] q
  );

  wire [25:0] accum_abs;
  wire [7:0] accum_rnd;

  reg signed [8:0] weights [INPUTS-1:0];
  reg signed [25:0] accum;
  reg signed [8:0] bias;

  reg [8:0] cnt;

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
      accum <= bias;
      cnt <= 0;
    end else if (en) begin
      if (cnt < INPUTS) begin
        cnt <= cnt + 5;
        accum <= ($signed({ 1'b0, d[0] }) * weights[cnt])     +
                 ($signed({ 1'b0, d[1] }) * weights[cnt + 1]) +
                 ($signed({ 1'b0, d[2] }) * weights[cnt + 2]) +
                 ($signed({ 1'b0, d[3] }) * weights[cnt + 3]) +
                 ($signed({ 1'b0, d[4] }) * weights[cnt + 4]);
      end
    end
  end

  assign accum_abs = accum[25] ? -accum : accum;

  assign accum_rnd = &accum_abs[24:10] ? 8'hFF : accum_abs[9:2];

  assign q[8] = accum[25];

endmodule
