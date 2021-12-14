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

  typedef enum logic [1:0] {
    IDLE = 2'b00,
    WAIT_NN = 2'b01,
    DONE = 2'b10
  } dnn_state;

  dnn_state state;

  reg [1:0] page_cnt;
  reg [10:0] cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      page_cnt <= '0;
      results <= '0;
      state <= IDLE;
      cnt <= '0;
    end else begin
      IDLE: if (prev_out_ready) begin
        state <= WAIT_NN;
      end
      WAIT_NN: if (det_nn_done) begin
        results[cnt] <= det_nn_result;
        if ((page_cnt < 2) && (cnt + 1 == 512)) begin
          page_cnt <= page_cnt + 1;
          state <= DONE;
          cnt <= '0;
        end else if ((page_cnt == 2) && (cnt + 1 == 62)) begin
          page_cnt <= '0;
          state <= DONE;
          cnt <= '0;
        end else begin
          cnt <= cnt + 1;
          state <= IDLE;
        end
      end
      DONE: if (next_in_ready) begin
        results <= '0;
        state <= IDLE;
      end
    end
  end

  assign in_ready = (state == IDLE);
  assign start_det_nn = (state == IDLE) & prev_out_ready;

  assign out_ready = (state == DONE);

endmodule
