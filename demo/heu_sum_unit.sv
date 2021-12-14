
/*
 * HEU sum unit
 */
module heu_sum_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input go,
    input [7:0] d [4:0],
    input [6:0] sums [4:0][255:0],

    /*
     * Outputs
     */
    output ready,
    output [7:0] q [4:0]
  );

  typedef enum reg [1:0] {
    IDLE = 2'b00,
    CALC = 2'b01,
    DONE = 2'b10
  } heu_sum_state;

  reg [8:0] cdist [255:0];
  reg [7:0] cnt;

  heu_sum_state state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (integer i = 0; i < 256; i++) begin
        cdist[i] <= 8'h00;
      end
      state <= IDLE;
      cnt <= 8'h00;
    end else begin
      case (state)
        IDLE: begin
          if (go) begin
            state <= CALC;
            cnt <= 8'h00;
          end
        end
        CALC: begin
          if (cnt > 0) begin
            cdist[cnt] <= cdist[cnt - 1] +
                            sums[0][cnt] +
                            sums[1][cnt] +
                            sums[2][cnt] +
                            sums[3][cnt] +
                            sums[4][cnt];
          end else begin
            cdist[0] <= sums[0][0] +
                        sums[1][0] +
                        sums[2][0] +
                        sums[3][0] +
                        sums[4][0];
          end
          if (cnt == 255) begin
            state <= DONE;
          end
          cnt <= cnt + 1;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  generate
    genvar i;
    for (i = 0; i < 5; i++) begin : sel
      heu_sum_lut lut (
        .d(cdist[d[i]]),
        .q(q[i])
      );
    end
  endgenerate

  assign ready = state == DONE;

endmodule
