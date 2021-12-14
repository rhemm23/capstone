module iru_ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input rnn_out_ready,
    input bcau_in_ready,

    /*
     * Outputs
     */
    output [4:0] row_sel [4:0],
    output [4:0] col_sel [4:0],
    output in_ready,
    output out_ready,
    output write_in,
    output rotate_in,
    output zero_out,
    output write_out
  );

  typedef enum logic[1:0] {
    IDLE = 2'b00,
    PROC = 2'b01,
    DONE = 2'b10
  } iru_state;

  iru_state state;

  reg [6:0] cnt;

  wire [7:0] base;
  wire [1:0] row_offset;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      cnt <= 0;
    end else begin
      case (state)
        IDLE: if (rnn_out_ready) begin
          state <= PROC;
        end
        PROC: begin
          if (cnt == 80) begin
            if (bcau_in_ready) begin
              state <= IDLE;
            end else begin
              state <= DONE;
            end
          end else begin
            cnt <= cnt + 1;
          end
        end
        DONE: if (bcau_in_ready) begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign in_ready = (state == IDLE);
  assign write_in = (state == IDLE) & rnn_out_ready;
  assign zero_out = (state == IDLE) & rnn_out_ready;

  assign rotate_in = (state == PROC) & (cnt < 80);
  assign write_out = (state == PROC) & (cnt < 80);

  assign out_ready = (state == DONE) |
                     (state == PROC & cnt == 80);

  assign base = (cnt < 20) ? 0  :
                (cnt < 40) ? 20 :
                (cnt < 60) ? 40 :
                60;

  assign row_offset = (cnt < 20) ? 0 :
                      (cnt < 40) ? 1 :
                      (cnt < 60) ? 2 :
                      3;

  generate
    genvar i;
    for (i = 0; i < 5; i++) begin : buffer_g
      assign row_sel[i] = (i * 4) + row_offset;
      assign col_sel[i] = cnt - base;
    end
  endgenerate

endmodule
