module a_neuron
  #(
    INPUTS = 401
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
    input [8:0] a_weight_sel,
    input signed [15:0] weights_d,
    input [7:0] d [4:0],

    /*
     * Outputs
     */
    output signed [15:0] q
  );

  wire [31:0] accum_abs;
  wire [7:0] accum_rnd;

  reg signed [15:0] weights [INPUTS-1:0]; // weights [400:1], bias [0]
  // reg signed [15:0] bias;
  reg signed [31:0] accum;
  reg [8:0] cnt;

  tanh_lut lut (
    .d(accum_rnd),
    .q(q[7:0])
  );

  // Different States
  typedef enum reg [1:0] {
    IDLE = 2'b00,
    LD_WEIGHTS = 2'b01,
    ACCUMULATE = 2'b10,
    EVALUATE = 2'b11
  } a_neuron_states;

  a_neuron_states state, nxt_state;

  // resets the state maching to IDLE and changes the state to nxt_state every clk cycle
  always_ff @(posedge clk, negedge rst_n) begin
      if(!rst_n)
      state <= IDLE;
      else
      state <= nxt_state;
  end
  

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      nxt_state <= state;
      weights <= '{ INPUTS { 0 } };
      accum <= 0;
      // bias <= 0;
      cnt <= 0;
    end else begin
      case (state)
        IDLE:
          if(wr_weights) begin
            weights[a_weight_sel] <= weights_d;
            nxt_state = LD_WEIGHTS;
          end else if (z) begin
            accum <= weights[0]; // weights[0] = bias
            cnt <= 0;
            nxt_state = ACCUMULATE;
          end
        LD_WEIGHTS:
          if(!wr_weights) begin
            nxt_state <= IDLE;
          end else begin
            weights[a_weight_sel] <= weights_d;
          end
        ACCUMULATE:
          if(en) begin
            if (cnt < (INPUTS-1)) begin
              cnt <= cnt + 5;
              accum <= accum +
                ($signed({ 8'h00, d[0] }) * weights[cnt + 1]) +
                ($signed({ 8'h00, d[1] }) * weights[cnt + 2]) +
                ($signed({ 8'h00, d[2] }) * weights[cnt + 3]) +
                ($signed({ 8'h00, d[3] }) * weights[cnt + 4]) +
                ($signed({ 8'h00, d[4] }) * weights[cnt + 5]);
            end
          end else begin
            nxt_state = IDLE;
          end
      endcase
    end
  end

  // TODO: NEED to FIX talk to ryan
  //assign accum_abs = accum[31] ? -accum : accum; // TODO: check correctness with test
  assign accum_abs = {1'b0, accum[30:0]};
  assign accum_rnd = (|accum_abs[30:10]) ? 8'hFF : accum_abs[9:2];

  assign q[8] = accum[31];

endmodule
