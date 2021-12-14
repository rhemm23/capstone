
/*
 * https://github.com/dawsonjon/fpu/blob/master/double_adder/double_adder.v
 */
module fp_adder
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

  reg [63:0] a, b, z;
  reg [55:0] a_m, b_m;
  reg [52:0] z_m;
  reg [12:0] a_e, b_e, z_e;
  reg a_s, b_s, z_s;
  reg guard, round_bit, sticky;
  reg [56:0] sum;

  typedef enum logic [3:0] {
    IDLE = 4'b0000,
    SPECIAL = 4'b0001,
    ALIGN = 4'b0010,
    ADD_0 = 4'b0011,
    ADD_1 = 4'b0100,
    NORM_1 = 4'b0101,
    NORM_2 = 4'b0110,
    ROUND = 4'b0111,
    PACK = 4'b1000,
    DONE = 4'b1001
  } adder_state;

  adder_state state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= SPECIAL;
          a <= a_in;
          b <= b_in;
          a_m <= { a_in[51:0], 3'b000 };
          b_m <= { b_in[51:0], 3'b000 };
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
            z[63] <= a_s;
            z[62:52] <= 2047;
            z[51:0] <= 0;
            if ((b_e == 1024) && (a_s != b_s)) begin
              z[63] <= 1;
              z[62:52] <= 2047;
              z[51] <= 1;
              z[50:0] <= 0;
            end
            state <= DONE;
          end else if (b_e == 1024) begin
            z[63] <= b_s;
            z[62:52] <= 2047;
            z[51:0] <= 0;
            state <= DONE;
          end else if ((($signed(a_e) == -1023) && (a_m == 0)) && (($signed(b_e) == -1023) && (b_m == 0))) begin
            z[63] <= a_s & b_s;
            z[62:52] <= b_e[10:0] + 1023;
            z[51:0] <= b_m[55:3];
            state <= DONE;
          end else if (($signed(a_e) == -1023) && (a_m == 0)) begin
            z[63] <= b_s;
            z[62:52] <= b_e[10:0] + 1023;
            z[51:0] <= b_m[55:3];
            state <= DONE;
          end else if (($signed(b_e) == -1023) && (b_m == 0)) begin
            z[63] <= a_s;
            z[62:52] <= a_e[10:0] + 1023;
            z[51:0] <= a_m[55:3];
            state <= DONE;
          end else begin
            if ($signed(a_e) == -1023) begin
              a_e <= -1022;
            end else begin
              a_m[55] <= 1;
            end
            if ($signed(b_e) == -1023) begin
              b_e <= -1022;
            end else begin
              b_m[55] <= 1;
            end
            state <= ALIGN;
          end
        end
        ALIGN: begin
          if ($signed(a_e) > $signed(b_e)) begin
            b_e <= b_e + 1;
            b_m <= b_m >> 1;
            b_m[0] <= b_m[0] | b_m[1];
          end else if ($signed(a_e) < $signed(b_e)) begin
            a_e <= a_e + 1;
            a_m <= a_m >> 1;
            a_m[0] <= a_m[0] | a_m[1];
          end else begin
            state <= ADD_0;
          end
        end
        ADD_0: begin
          z_e <= a_e;
          if (a_s == b_s) begin
            sum <= { 1'b0, a_m } + b_m;
            z_s <= a_s;
          end else begin
            if (a_m > b_m) begin
              sum <= { 1'b0, a_m } - b_m;
              z_s <= a_s;
            end else begin
              sum <= { 1'b0, b_m } - a_m;
              z_s <= b_s;
            end
          end
          state <= ADD_1;
        end
        ADD_1: begin
          if (sum[56]) begin
            z_m <= sum[56:4];
            guard <= sum[3];
            round_bit <= sum[2];
            sticky <= sum[1] | sum[0];
            z_e <= z_e + 1;
          end else begin
            z_m <= sum[55:3];
            guard <= sum[2];
            round_bit <= sum[1];
            sticky <= sum[0];
          end
          state <= NORM_1;
        end
        NORM_1: begin
          if (z_m[52] == 0 && $signed(z_e) > -1022) begin
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
          z[62:52] <= z_e[10:0] + 1023;
          z[63] <= z_s;
          if ($signed(z_e) == -1022 && z_m[52] == 0) begin
            z[62:52] <= 0;
          end
          if ($signed(z_e) == -1022 && z_m[52:0] == 0) begin
            z[63] <= 0;
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
