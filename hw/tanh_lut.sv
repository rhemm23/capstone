module tanh_lut
  (
    /*
     * Inputs
     */
    input [7:0] d, // [3-int][5-fraction]

    /*
     * Outputs
     */
    output reg [7:0] q // [1-int][7-fraction]
  );

  always_comb begin
    case (d)
      8'b00000000: q <= 8'b00000000;
      8'b00000001: q <= 8'b00000011;
      8'b00000010: q <= 8'b00000111;
      8'b00000011: q <= 8'b00001011;
      8'b00000100: q <= 8'b00001111;
      8'b00000101: q <= 8'b00010011;
      8'b00000110: q <= 8'b00010111;
      8'b00000111: q <= 8'b00011011;
      8'b00001000: q <= 8'b00011111;
      8'b00001001: q <= 8'b00100011;
      8'b00001010: q <= 8'b00100110;
      8'b00001011: q <= 8'b00101010;
      8'b00001100: q <= 8'b00101101;
      8'b00001101: q <= 8'b00110001;
      8'b00001110: q <= 8'b00110100;
      8'b00001111: q <= 8'b00110111;
      8'b00010000: q <= 8'b00111011;
      8'b00010001: q <= 8'b00111110;
      8'b00010010: q <= 8'b01000001;
      8'b00010011: q <= 8'b01000100;
      8'b00010100: q <= 8'b01000110;
      8'b00010101: q <= 8'b01001001;
      8'b00010110: q <= 8'b01001100;
      8'b00010111: q <= 8'b01001110;
      8'b00011000: q <= 8'b01010001;
      8'b00011001: q <= 8'b01010011;
      8'b00011010: q <= 8'b01010101;
      8'b00011011: q <= 8'b01011000;
      8'b00011100: q <= 8'b01011010;
      8'b00011101: q <= 8'b01011100;
      8'b00011110: q <= 8'b01011101;
      8'b00011111: q <= 8'b01011111;
      8'b00100000: q <= 8'b01100001;
      8'b00100001: q <= 8'b01100011;
      8'b00100010: q <= 8'b01100100;
      8'b00100011: q <= 8'b01100110;
      8'b00100100: q <= 8'b01100111;
      8'b00100101: q <= 8'b01101000;
      8'b00100110: q <= 8'b01101010;
      8'b00100111: q <= 8'b01101011;
      8'b00101000: q <= 8'b01101100;
      8'b00101001: q <= 8'b01101101;
      8'b00101010: q <= 8'b01101110;
      8'b00101011: q <= 8'b01101111;
      8'b00101100: q <= 8'b01110000;
      8'b00101101: q <= 8'b01110001;
      8'b00101110: q <= 8'b01110010;
      8'b00101111: q <= 8'b01110011;
      8'b00110000: q <= 8'b01110011;
      8'b00110001: q <= 8'b01110100;
      8'b00110010: q <= 8'b01110101;
      8'b00110011: q <= 8'b01110101;
      8'b00110100: q <= 8'b01110110;
      8'b00110101: q <= 8'b01110111;
      8'b00110110: q <= 8'b01110111;
      8'b00110111: q <= 8'b01111000;
      8'b00111000: q <= 8'b01111000;
      8'b00111001: q <= 8'b01111000;
      8'b00111010: q <= 8'b01111001;
      8'b00111011: q <= 8'b01111001;
      8'b00111100: q <= 8'b01111010;
      8'b00111101: q <= 8'b01111010;
      8'b00111110: q <= 8'b01111010;
      8'b00111111: q <= 8'b01111011;
      8'b01000000: q <= 8'b01111011;
      8'b01000001: q <= 8'b01111011;
      8'b01000010: q <= 8'b01111011;
      8'b01000011: q <= 8'b01111100;
      8'b01000100: q <= 8'b01111100;
      8'b01000101: q <= 8'b01111100;
      8'b01000110: q <= 8'b01111100;
      8'b01000111: q <= 8'b01111101;
      8'b01001000: q <= 8'b01111101;
      8'b01001001: q <= 8'b01111101;
      8'b01001010: q <= 8'b01111101;
      8'b01001011: q <= 8'b01111101;
      8'b01001100: q <= 8'b01111101;
      8'b01001101: q <= 8'b01111101;
      8'b01001110: q <= 8'b01111110;
      8'b01001111: q <= 8'b01111110;
      8'b01010000: q <= 8'b01111110;
      8'b01010001: q <= 8'b01111110;
      8'b01010010: q <= 8'b01111110;
      8'b01010011: q <= 8'b01111110;
      8'b01010100: q <= 8'b01111110;
      8'b01010101: q <= 8'b01111110;
      8'b01010110: q <= 8'b01111110;
      8'b01010111: q <= 8'b01111110;
      8'b01011000: q <= 8'b01111110;
      8'b01011001: q <= 8'b01111111;
      8'b01011010: q <= 8'b01111111;
      8'b01011011: q <= 8'b01111111;
      8'b01011100: q <= 8'b01111111;
      8'b01011101: q <= 8'b01111111;
      8'b01011110: q <= 8'b01111111;
      8'b01011111: q <= 8'b01111111;
      8'b01100000: q <= 8'b01111111;
      8'b01100001: q <= 8'b01111111;
      8'b01100010: q <= 8'b01111111;
      8'b01100011: q <= 8'b01111111;
      8'b01100100: q <= 8'b01111111;
      8'b01100101: q <= 8'b01111111;
      8'b01100110: q <= 8'b01111111;
      8'b01100111: q <= 8'b01111111;
      8'b01101000: q <= 8'b01111111;
      8'b01101001: q <= 8'b01111111;
      8'b01101010: q <= 8'b01111111;
      8'b01101011: q <= 8'b01111111;
      8'b01101100: q <= 8'b01111111;
      8'b01101101: q <= 8'b01111111;
      8'b01101110: q <= 8'b01111111;
      8'b01101111: q <= 8'b01111111;
      8'b01110000: q <= 8'b01111111;
      8'b01110001: q <= 8'b01111111;
      8'b01110010: q <= 8'b01111111;
      8'b01110011: q <= 8'b01111111;
      8'b01110100: q <= 8'b01111111;
      8'b01110101: q <= 8'b01111111;
      8'b01110110: q <= 8'b01111111;
      8'b01110111: q <= 8'b01111111;
      8'b01111000: q <= 8'b01111111;
      8'b01111001: q <= 8'b01111111;
      8'b01111010: q <= 8'b01111111;
      8'b01111011: q <= 8'b01111111;
      8'b01111100: q <= 8'b01111111;
      8'b01111101: q <= 8'b01111111;
      8'b01111110: q <= 8'b01111111;
      8'b01111111: q <= 8'b01111111;
      8'b10000000: q <= 8'b01111111;
      8'b10000001: q <= 8'b01111111;
      8'b10000010: q <= 8'b01111111;
      8'b10000011: q <= 8'b01111111;
      8'b10000100: q <= 8'b01111111;
      8'b10000101: q <= 8'b01111111;
      8'b10000110: q <= 8'b01111111;
      8'b10000111: q <= 8'b01111111;
      8'b10001000: q <= 8'b01111111;
      8'b10001001: q <= 8'b01111111;
      8'b10001010: q <= 8'b01111111;
      8'b10001011: q <= 8'b01111111;
      8'b10001100: q <= 8'b01111111;
      8'b10001101: q <= 8'b01111111;
      8'b10001110: q <= 8'b01111111;
      8'b10001111: q <= 8'b01111111;
      8'b10010000: q <= 8'b01111111;
      8'b10010001: q <= 8'b01111111;
      8'b10010010: q <= 8'b01111111;
      8'b10010011: q <= 8'b01111111;
      8'b10010100: q <= 8'b01111111;
      8'b10010101: q <= 8'b01111111;
      8'b10010110: q <= 8'b01111111;
      8'b10010111: q <= 8'b01111111;
      8'b10011000: q <= 8'b01111111;
      8'b10011001: q <= 8'b01111111;
      8'b10011010: q <= 8'b01111111;
      8'b10011011: q <= 8'b01111111;
      8'b10011100: q <= 8'b01111111;
      8'b10011101: q <= 8'b01111111;
      8'b10011110: q <= 8'b01111111;
      8'b10011111: q <= 8'b01111111;
      8'b10100000: q <= 8'b01111111;
      8'b10100001: q <= 8'b01111111;
      8'b10100010: q <= 8'b01111111;
      8'b10100011: q <= 8'b01111111;
      8'b10100100: q <= 8'b01111111;
      8'b10100101: q <= 8'b01111111;
      8'b10100110: q <= 8'b01111111;
      8'b10100111: q <= 8'b01111111;
      8'b10101000: q <= 8'b01111111;
      8'b10101001: q <= 8'b01111111;
      8'b10101010: q <= 8'b01111111;
      8'b10101011: q <= 8'b01111111;
      8'b10101100: q <= 8'b01111111;
      8'b10101101: q <= 8'b01111111;
      8'b10101110: q <= 8'b01111111;
      8'b10101111: q <= 8'b01111111;
      8'b10110000: q <= 8'b01111111;
      8'b10110001: q <= 8'b01111111;
      8'b10110010: q <= 8'b01111111;
      8'b10110011: q <= 8'b01111111;
      8'b10110100: q <= 8'b01111111;
      8'b10110101: q <= 8'b01111111;
      8'b10110110: q <= 8'b01111111;
      8'b10110111: q <= 8'b01111111;
      8'b10111000: q <= 8'b01111111;
      8'b10111001: q <= 8'b01111111;
      8'b10111010: q <= 8'b01111111;
      8'b10111011: q <= 8'b01111111;
      8'b10111100: q <= 8'b01111111;
      8'b10111101: q <= 8'b01111111;
      8'b10111110: q <= 8'b01111111;
      8'b10111111: q <= 8'b01111111;
      8'b11000000: q <= 8'b01111111;
      8'b11000001: q <= 8'b01111111;
      8'b11000010: q <= 8'b01111111;
      8'b11000011: q <= 8'b01111111;
      8'b11000100: q <= 8'b01111111;
      8'b11000101: q <= 8'b01111111;
      8'b11000110: q <= 8'b01111111;
      8'b11000111: q <= 8'b01111111;
      8'b11001000: q <= 8'b01111111;
      8'b11001001: q <= 8'b01111111;
      8'b11001010: q <= 8'b01111111;
      8'b11001011: q <= 8'b01111111;
      8'b11001100: q <= 8'b01111111;
      8'b11001101: q <= 8'b01111111;
      8'b11001110: q <= 8'b01111111;
      8'b11001111: q <= 8'b01111111;
      8'b11010000: q <= 8'b01111111;
      8'b11010001: q <= 8'b01111111;
      8'b11010010: q <= 8'b01111111;
      8'b11010011: q <= 8'b01111111;
      8'b11010100: q <= 8'b01111111;
      8'b11010101: q <= 8'b01111111;
      8'b11010110: q <= 8'b01111111;
      8'b11010111: q <= 8'b01111111;
      8'b11011000: q <= 8'b01111111;
      8'b11011001: q <= 8'b01111111;
      8'b11011010: q <= 8'b01111111;
      8'b11011011: q <= 8'b01111111;
      8'b11011100: q <= 8'b01111111;
      8'b11011101: q <= 8'b01111111;
      8'b11011110: q <= 8'b01111111;
      8'b11011111: q <= 8'b01111111;
      8'b11100000: q <= 8'b01111111;
      8'b11100001: q <= 8'b01111111;
      8'b11100010: q <= 8'b01111111;
      8'b11100011: q <= 8'b01111111;
      8'b11100100: q <= 8'b01111111;
      8'b11100101: q <= 8'b01111111;
      8'b11100110: q <= 8'b01111111;
      8'b11100111: q <= 8'b01111111;
      8'b11101000: q <= 8'b01111111;
      8'b11101001: q <= 8'b01111111;
      8'b11101010: q <= 8'b01111111;
      8'b11101011: q <= 8'b01111111;
      8'b11101100: q <= 8'b01111111;
      8'b11101101: q <= 8'b01111111;
      8'b11101110: q <= 8'b01111111;
      8'b11101111: q <= 8'b01111111;
      8'b11110000: q <= 8'b01111111;
      8'b11110001: q <= 8'b01111111;
      8'b11110010: q <= 8'b01111111;
      8'b11110011: q <= 8'b01111111;
      8'b11110100: q <= 8'b01111111;
      8'b11110101: q <= 8'b01111111;
      8'b11110110: q <= 8'b01111111;
      8'b11110111: q <= 8'b01111111;
      8'b11111000: q <= 8'b01111111;
      8'b11111001: q <= 8'b01111111;
      8'b11111010: q <= 8'b01111111;
      8'b11111011: q <= 8'b01111111;
      8'b11111100: q <= 8'b01111111;
      8'b11111101: q <= 8'b01111111;
      8'b11111110: q <= 8'b01111111;
      8'b11111111: q <= 8'b01111111;
      default: q <= 0;
    endcase
  end

endmodule
