module iru_sin_lut
  (
    /*
     * Inputs
     */
    input [35:0] d,

    /*
     * Outputs
     */
    output reg [8:0] q
  );

  always_comb begin
    case (d)
      36'b000000000000000000000000000000000001: q <= 8'b000000000;
      36'b000000000000000000000000000000000010: q <= 8'b000010110;
      36'b000000000000000000000000000000000100: q <= 8'b000101011;
      36'b000000000000000000000000000000001000: q <= 8'b000111111;
      36'b000000000000000000000000000000010000: q <= 8'b001010010;
      36'b000000000000000000000000000000100000: q <= 8'b001100010;
      36'b000000000000000000000000000001000000: q <= 8'b001101110;
      36'b000000000000000000000000000010000000: q <= 8'b001111000;
      36'b000000000000000000000000000100000000: q <= 8'b001111110;
      36'b000000000000000000000000001000000000: q <= 8'b010000000;
      36'b000000000000000000000000010000000000: q <= 8'b001111110;
      36'b000000000000000000000000100000000000: q <= 8'b001111000;
      36'b000000000000000000000001000000000000: q <= 8'b001101110;
      36'b000000000000000000000010000000000000: q <= 8'b001100010;
      36'b000000000000000000000100000000000000: q <= 8'b001010010;
      36'b000000000000000000001000000000000000: q <= 8'b000111111;
      36'b000000000000000000010000000000000000: q <= 8'b000101011;
      36'b000000000000000000100000000000000000: q <= 8'b000010110;
      36'b000000000000000001000000000000000000: q <= 8'b000000000;
      36'b000000000000000010000000000000000000: q <= 8'b111101010;
      36'b000000000000000100000000000000000000: q <= 8'b111010101;
      36'b000000000000001000000000000000000000: q <= 8'b111000000;
      36'b000000000000010000000000000000000000: q <= 8'b110101110;
      36'b000000000000100000000000000000000000: q <= 8'b110011110;
      36'b000000000001000000000000000000000000: q <= 8'b110010010;
      36'b000000000010000000000000000000000000: q <= 8'b110001000;
      36'b000000000100000000000000000000000000: q <= 8'b110000010;
      36'b000000001000000000000000000000000000: q <= 8'b110000000;
      36'b000000010000000000000000000000000000: q <= 8'b110000010;
      36'b000000100000000000000000000000000000: q <= 8'b110001000;
      36'b000001000000000000000000000000000000: q <= 8'b110010010;
      36'b000010000000000000000000000000000000: q <= 8'b110011110;
      36'b000100000000000000000000000000000000: q <= 8'b110101110;
      36'b001000000000000000000000000000000000: q <= 8'b111000000;
      36'b010000000000000000000000000000000000: q <= 8'b111010101;
      36'b100000000000000000000000000000000000: q <= 8'b111101010;
      default: q <= 0;
    endcase
  end

endmodule
