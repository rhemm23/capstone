module rdn_ctrl_unit_fp
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
    output logic start_rnn,
    output logic in_ready,
    output logic out_ready
  );

  typedef enum reg [1:0] {
    IDLE = 2'b00,
    WR_WEIGHTS = 2'b01,
    EXECUTING = 2'b10,
    DONE = 2'b11
  } rdn_state;

  rdn_state state, nxt_state;
  reg weights_loaded;

  // Indicates if the weights have been loaded.
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
      weights_loaded <= 1'b0;
    end else if (weight_valid) begin
      weights_loaded <= 1'b1;
    end
  end

  // resets the state maching to IDLE and changes the state to nxt_state every clk cycle
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
      state <= IDLE;
    else
      state <= nxt_state;
  end

  always_comb begin
    nxt_state = state;
    in_ready = 1'b0;
    start_rnn = 1'b0;
    out_ready = 1'b0;
    case (state)
      IDLE: 
        if (heu_out_valid && weights_loaded) begin
          start_rnn = 1'b1;
          nxt_state = EXECUTING;
        end else if (load_weights) begin
          nxt_state = WR_WEIGHTS;
        end else begin
          in_ready = 1'b1;
        end
      WR_WEIGHTS: 
        if (weight_valid) begin
          nxt_state = IDLE;
        end
      EXECUTING: 
        if (rot_nn_done) begin
          out_ready = 1'b1;
          nxt_state = DONE;
        end
      DONE: 
        if (iru_in_ready) begin
          in_ready = 1'b1;
          nxt_state = IDLE;
        end else begin
          out_ready = 1'b1;
        end
    endcase
  end
  

endmodule
