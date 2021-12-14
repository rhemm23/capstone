
/*
 * https://github.com/dawsonjon/fpu/blob/master/double_multiplier/double_multiplier.v
 */
module fp_mult
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input [63:0] a_in,
    input [63:0] b_in,
    input start,

    /*
     * Outputs
     */
    output [63:0] c_out,
    output done
  );

  typedef enum logic [3:0] {
    IDLE = 4'b0000,
    SPECIAL = 4'b0001,
    NORM_A = 4'b0010,
    NORM_B = 4'b0011,
    MULT_0 = 4'b0100,
    MULT_1 = 4'b0101,
    NORM_1 = 4'b0110,
    NORM_2 = 4'b0111,
    ROUND = 4'b1000,
    PACK = 4'b1001,
    DONE = 4'b1010
  } mult_state;

  mult_state state;

  reg [63:0] a, b, z;
  reg [52:0] a_m, b_m, z_m;
  reg [12:0] a_e, b_e, z_e;
  reg a_s, b_s, z_s;
  reg guard, round_bit, sticky;
  reg [105:0] product;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= SPECIAL;
          a <= a_in;
          b <= b_in;
          a_m <= a_in[51:0];
          b_m <= b_in[51:0];
          a_e <= a_in[62:52] - 1023;
          b_e <= b_in[62:52] - 1023;
          a_s <= a_in[63];
          b_s <= b_in[63];
        end
        SPECIAL: begin
          if ((a_e == 1024 && a_m != 0) || (b_e == 1024 && b_m != 0)) begin
            z[63] <= 1;
            z[62:52] <= 2047;
            z[51] <= 1;
            z[50:0] <= 0;
            state <= DONE;
          end else if (a_e == 1024) begin
            z[63] <= a_s ^ b_s;
            z[62:52] <= 2047;
            z[51:0] <= 0;
            state <= DONE;
            if (($signed(b_e) == -1023) && (b_m == 0)) begin
              z[63] <= 1;
              z[62:52] <= 2047;
              z[51] <= 1;
              z[50:0] <= 0;
              state <= DONE;
            end
          end else if (b_e == 1024) begin
            z[63] <= a_s ^ b_s;
            z[62:52] <= 2047;
            z[51:0] <= 0;
            if (($signed(a_e) == -1023) && (a_m == 0)) begin
              z[63] <= 1;
              z[62:52] <= 2047;
              z[51] <= 1;
              z[50:0] <= 0;
              state <= DONE;
            end
            state <= DONE;
          end else if (($signed(a_e) == -1023) && (a_m == 0)) begin
            z[63] <= a_s ^ b_s;
            z[62:52] <= 0;
            z[51:0] <= 0;
            state <= DONE;
          end else if (($signed(b_e) == -1023) && (b_m == 0)) begin
            z[63] <= a_s ^ b_s;
            z[62:52] <= 0;
            z[51:0] <= 0;
            state <= DONE;
          end else begin
            if ($signed(a_e) == -1023) begin
              a_e <= -1022;
            end else begin
              a_m[52] <= 1;
            end
            if ($signed(b_e) == -1023) begin
              b_e <= -1022;
            end else begin
              b_m[52] <= 1;
            end
            state <= NORM_A;
          end
        end
        NORM_A: begin
          if (a_m[52]) begin
            state <= NORM_B;
          end else begin
            a_m <= a_m << 1;
            a_e <= a_e - 1;
          end
        end
        NORM_B: begin
          if (b_m[52]) begin
            state <= MULT_0;
          end else begin
            b_m <= b_m << 1;
            b_e <= b_e - 1;
          end
        end
        MULT_0: begin
          z_s <= a_s ^ b_s;
          z_e <= a_e + b_e + 1;
          product <= a_m * b_m;
          state <= MULT_1;
        end
        MULT_1: begin
          z_m <= product[105:53];
          guard <= product[52];
          round_bit <= product[51];
          sticky <= (product[50:0] != 0);
          state <= NORM_1;
        end
        NORM_1: begin
          if (z_m[52] == 0) begin
            z_e <= z_e - 1;
            z_m <= z_m << 1;
            z_m[0] <= guard;
            guard <= round_bit;
            round_bit <= 0;
          end else begin
            state <= NORM_2;
          end
        end
        NORM_2: begin
          if ($signed(z_e) < -1022) begin
            z_e <= z_e + 1;
            z_m <= z_m >> 1;
            guard <= z_m[0];
            round_bit <= guard;
            sticky <= sticky | round_bit;
          end else begin
            state <= ROUND;
          end
        end
        ROUND: begin
          if (guard && (round_bit | sticky | z_m[0])) begin
            z_m <= z_m + 1;
            if (z_m == 53'h1fffffffffffff) begin
              z_e <= z_e + 1;
            end
          end
          state <= PACK;
        end
        PACK: begin
          z[51:0] <= z_m[51:0];
          z[62:52] <= z_e[11:0] + 1023;
          z[63] <= z_s;
          if ($signed(z_e) == -1022 && z_m[52] == 0) begin
            z[62:52] <= 0;
          end
          if ($signed(z_e) > 1023) begin
            z[51:0] <= 0;
            z[62:52] <= 2047;
            z[63] <= z_s;
          end
          state <= DONE;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign c_out = z;
  assign done = (state == DONE);

endmodule
