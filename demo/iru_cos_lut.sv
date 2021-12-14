module iru_cos_lut
  (
    /*
     * Inputs
     */
    input [35:0] d,

    /*
     * Outputs
     */
    output reg [31:0] q
  );

  always_comb begin
    case (d)
      36'b000000000000000000000000000000000001: q <= 32'b01000000000000000000000000000000;
      36'b000000000000000000000000000000000010: q <= 32'b00111111000001110001011100011001;
      36'b000000000000000000000000000000000100: q <= 32'b00111100001000111110110010000100;
      36'b000000000000000000000000000000001000: q <= 32'b00110111011011001111010111010000;
      36'b000000000000000000000000000000010000: q <= 32'b00110001000001101101111101000101;
      36'b000000000000000000000000000000100000: q <= 32'b00101001001000110110111010100100;
      36'b000000000000000000000000000001000000: q <= 32'b00100000000000000000000000000000;
      36'b000000000000000000000000000010000000: q <= 32'b00010101111000111010100001110100;
      36'b000000000000000000000000000100000000: q <= 32'b00001011000111010000110100111111;
      36'b000000000000000000000000001000000000: q <= 32'b00000000000000000000000000000000;
      36'b000000000000000000000000010000000000: q <= 32'b11110100111000101111001011000001;
      36'b000000000000000000000000100000000000: q <= 32'b11101010000111000101011110001100;
      36'b000000000000000000000001000000000000: q <= 32'b11100000000000000000000000000001;
      36'b000000000000000000000010000000000000: q <= 32'b11010110110111001001000101011100;
      36'b000000000000000000000100000000000000: q <= 32'b11001110111110010010000010111011;
      36'b000000000000000000001000000000000000: q <= 32'b11001000100100110000101000110000;
      36'b000000000000000000010000000000000000: q <= 32'b11000011110111000001001101111100;
      36'b000000000000000000100000000000000000: q <= 32'b11000000111110001110100011100111;
      36'b000000000000000001000000000000000000: q <= 32'b11000000000000000000000000000000;
      36'b000000000000000010000000000000000000: q <= 32'b11000000111110001110100011100111;
      36'b000000000000000100000000000000000000: q <= 32'b11000011110111000001001101111100;
      36'b000000000000001000000000000000000000: q <= 32'b11001000100100110000101000110000;
      36'b000000000000010000000000000000000000: q <= 32'b11001110111110010010000010111011;
      36'b000000000000100000000000000000000000: q <= 32'b11010110110111001001000101011100;
      36'b000000000001000000000000000000000000: q <= 32'b11100000000000000000000000000000;
      36'b000000000010000000000000000000000000: q <= 32'b11101010000111000101011110001100;
      36'b000000000100000000000000000000000000: q <= 32'b11110100111000101111001011000001;
      36'b000000001000000000000000000000000000: q <= 32'b00000000000000000000000000000000;
      36'b000000010000000000000000000000000000: q <= 32'b00001011000111010000110100111111;
      36'b000000100000000000000000000000000000: q <= 32'b00010101111000111010100001110100;
      36'b000001000000000000000000000000000000: q <= 32'b00100000000000000000000000000000;
      36'b000010000000000000000000000000000000: q <= 32'b00101001001000110110111010100100;
      36'b000100000000000000000000000000000000: q <= 32'b00110001000001101101111101000101;
      36'b001000000000000000000000000000000000: q <= 32'b00110111011011001111010111010000;
      36'b010000000000000000000000000000000000: q <= 32'b00111100001000111110110010000100;
      36'b100000000000000000000000000000000000: q <= 32'b00111111000001110001011100011001;
      default: q <= '0;
    endcase
  end

endmodule
