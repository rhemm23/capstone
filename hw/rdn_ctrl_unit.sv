module rdn_ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input heu_out_ready,
    input iru_in_ready,
    input [8:0] a_layer_out [14:0],
    input [8:0] b_layer_out [14:0],

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

  typedef enum reg [2:0] {
    IDLE = 3'b000,
    A_CALC = 3'b001,
    B_CALC = 3'b010,
    C_CALC = 3'b011,
    DONE = 3'b100
  } rdn_state;

  rdn_state state;

  reg [6:0] cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (heu_out_ready) begin
          state <= A_CALC;
          cnt <= 7'h00;
        end
        A_CALC: begin
          if (cnt == 80) begin
            state <= B_CALC;
            cnt <= 7'h00;
          end else begin
            cnt <= cnt + 1;
          end
        end
        B_CALC: begin
          if (cnt == 15) begin
            state <= C_CALC;
            cnt <= 7'h00;
          end else begin
            cnt <= cnt + 1;
          end
        end
        C_CALC: begin
          if (cnt == 15) begin
            if (iru_in_ready) begin
              state <= IDLE;
            end else begin
              state <= DONE;
            end
          end else begin
            cnt <= cnt + 1;
          end
        end
        DONE: if (iru_in_ready) begin
          state <= IDLE;
        end
      endcase
    end
  end

  /*
   * IDLE -> ANY
   */
  assign in_ready = (state == IDLE);

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
