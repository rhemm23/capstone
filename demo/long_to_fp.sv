
/*
 * https://github.com/dawsonjon/fpu/blob/master/long_to_double/long_to_double.v
 */
module long_to_fp
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input [63:0] long_in,
    input start,

    /*
     * Outputs
     */
    output [63:0] fp_out,
    output done
  );

  reg [63:0] a, z, value;
  reg [52:0] z_m;
  reg [10:0] z_r;
  reg [10:0] z_e;
  reg z_s;
  reg guard, round_bit, sticky;

  typedef enum logic [2:0] {
    IDLE = 3'b000,
    CONV_0 = 3'b001,
    CONV_1 = 3'b010,
    CONV_2 = 3'b011,
    ROUND = 3'b100,
    PACK = 3'b101,
    DONE = 3'b110
  } l2fp_state;

  l2fp_state state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= CONV_0;
          a <= long_in;
        end
        CONV_0: begin
          if (a == 0) begin
            z_s <= 0;
            z_m <= 0;
            z_e <= -1023;
            state <= PACK;
          end else begin
            value <= a[63] ? -a : a;
            z_s <= a[63];
            state <= CONV_1;
          end
        end
        CONV_1: begin
           z_e <= 63;
          z_m <= value[63:11];
          z_r <= value[10:0];
          state <= CONV_2;
        end
        CONV_2: begin
          if (!z_m[52]) begin
            z_e <= z_e - 1;
            z_m <= z_m << 1;
            z_m[0] <= z_r[10];
            z_r <= z_r << 1;
          end else begin
            guard <= z_r[10];
            round_bit <= z_r[9];
            sticky <= z_r[8:0] != 0;
            state <= ROUND;
          end
        end
        ROUND: begin
          if (guard && (round_bit || sticky || z_m[0])) begin
            z_m <= z_m + 1;
            if (z_m == 53'h1fffffffffffff) begin
              z_e <= z_e + 1;
            end
          end
          state <= PACK;
        end
        PACK: begin
          z[51:0] <= z_m[51:0];
          z[62:52] <= z_e + 1023;
          z[63] <= z_s;
          state <= DONE;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign fp_out = z;
  assign done = (state == DONE);

endmodule
