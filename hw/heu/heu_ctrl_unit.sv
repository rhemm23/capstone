module heu_ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input sum_ready,
    input next_in_ready,
    input prev_out_ready,

    /*
     * Outputs
     */
    output sum_go,
    output in_ready,
    output write_in,
    output out_ready,
    output rotate_in,
    output shift_out,
    output zero_cnts,
    output enable_calc
  );

  typedef enum reg [2:0] {
    IDLE = 3'b000,
    CALC = 3'b001,
    DO_SUM = 3'b010,
    MOVE = 3'b011,
    DONE = 3'b100
  } heu_state;

  heu_state state;

  reg [6:0] cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      cnt <= 7'h00;
    end else begin
      case (state)
        IDLE: if (prev_out_ready) begin
          state <= CALC;
          cnt <= 7'h00;
        end
        CALC: begin
          if (cnt == 80) begin
            state <= DO_SUM;
          end else begin
            cnt <= cnt + 1;
          end
        end
        DO_SUM: if (sum_ready) begin
          state <= MOVE;
          cnt <= 7'h00;
        end
        MOVE: begin
          if (cnt == 80) begin
            if (next_in_ready) begin
              state <= IDLE;
            end else begin
              state <= DONE;
            end
          end else begin
            cnt <= cnt + 1;
          end
        end
        DONE: if (next_in_ready) begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign in_ready = (state == IDLE);
  assign write_in = (state == IDLE) & prev_out_ready;
  assign zero_cnts = (state == IDLE) & prev_out_ready;

  assign sum_go = (state == CALC) & (cnt == 80);
  assign rotate_in = ((state == CALC) & (cnt < 80)) |
                     ((state == MOVE) & (cnt < 80));
  assign enable_calc = (state == CALC) & (cnt < 80);

  assign shift_out = (state == MOVE) & (cnt < 80);
  assign out_ready = (state == DONE) | (state == MOVE & cnt == 80);

endmodule
