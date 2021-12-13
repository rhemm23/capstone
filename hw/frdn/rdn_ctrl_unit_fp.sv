module rdn_ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input heu_out_valid,
    input iru_in_ready,
    input load_weights,
    input weight_valid,
    input rot_nn_done,

    /*
     * Outputs
     */
    output rotate_in,
    output write_in,
    output shift_out,
    output z_a_layer,
    output z_b_layer,
    output z_c_layer,
    output en_a_layer,
    output en_b_layer,
    output en_c_layer,
    output [8:0] b_layer_in,
    output [8:0] c_layer_in,
    output in_ready,
    output out_ready
  );

  typedef enum reg [1:0] {
    IDLE = 2'b00,
    WR_WEIGHTS = 2'b01,
    EXECUTING = 2'b10,
    DONE = 2'b11
  } rdn_state;

  rdn_state state;
  reg weights_loaded;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      weights_loaded <= 1'b0;
    end else begin
      case (state)
        IDLE: 
          if (heu_out_valid && weights_loaded) begin
            state <= EXECUTING;
          end else if (load_weights) begin
            state <= WR_WEIGHTS;
          end else begin
            state <= state;
          end
        WR_WEIGHTS: 
          if (weight_valid) begin
            state <= IDLE;
            weights_loaded <= 1'b1;
          end else begin
            state <= state;
          end
        EXECUTING: 
          if (rot_nn_done) begin
            state <= DONE;
            cnt <= 7'h00;
          end else begin
            state <= state;
          end
        DONE: 
          if (iru_in_ready) begin
            state <= IDLE;
          end else begin
            state <= state;
          end
      endcase
    end
  end

  /*
   * IDLE -> ANY
   */
  assign in_ready = ((state == IDLE) && weights_loaded);

  /*
   * IDLE -> A_CALC
   */
  assign write_in = (state == IDLE) & heu_out_ready;
  assign z_a_layer = (state == IDLE) & heu_out_ready;

  /*
   * A_CALC -> A_CALC
   */
  assign shift_out = (state == A_CALC) & (cnt < 80);
  assign rotate_in = (state == A_CALC) & (cnt < 80);
  assign en_a_layer = (state == A_CALC) & (cnt < 80);

  /*
   * A_CALC -> B_CALC
   */
  assign z_b_layer = (state == A_CALC) & (cnt == 80);

  /*
   * B_CALC -> B_CALC
   */
  assign en_b_layer = (state == B_CALC) & (cnt < 15);

  /*
   * B_CALC -> C_CALC
   */
  assign z_c_layer = (state == B_CALC) & (cnt == 15);

  /*
   * C_CALC -> C_CALC
   */
  assign en_c_layer = (state == C_CALC) & (cnt < 15);

  /*
   * C_CALC -> ANY | DONE -> ANY
   */
  assign out_ready = (state == DONE) | (state == C_CALC & cnt == 15);

  assign b_layer_in = (state == B_CALC & cnt < 15) ? a_layer_out[cnt] : 0;
  assign c_layer_in = (state == C_CALC & cnt < 15) ? b_layer_out[cnt] : 0;

endmodule
