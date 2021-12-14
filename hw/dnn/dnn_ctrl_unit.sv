module dnn_ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input det_nn_done,
    input det_nn_result,
    input prev_out_ready,
    input next_in_ready,

    /*
     * Outputs
     */
    output in_ready,
    output out_ready,
    output start_det_nn,
    output reg [511:0] results
  );

  typedef enum logic [] {
    IDLE,
    WAIT_NN,
    DONE
  } dnn_state;

  dnn_state state;

  reg [10:0] cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      results <= '0;
      state <= IDLE;
      cnt <= '0;
    end else begin
      IDLE: if (prev_out_ready) begin
        state <= WAIT_NN;
      end
      WAIT_NN: if (det_nn_done) begin
        
      end
    end
  end

  assign in_ready = (state == IDLE);
  assign start_det_nn = (state == IDLE) & prev_out_ready;

  assign out_ready = (state == );

endmodule
