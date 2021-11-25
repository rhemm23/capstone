import math

# Open tanh lut verilog file
with open('../hw/tanh_lut.sv', 'w+') as lut:
  lut.write('module tanh_lut\n')
  lut.write('  (\n')
  lut.write('    /*\n')
  lut.write('     * Inputs\n')
  lut.write('     */\n')
  lut.write('    input [7:0] d,\n\n')
  lut.write('    /*\n')
  lut.write('     * Outputs\n')
  lut.write('     */\n')
  lut.write('    output reg [7:0] q\n')
  lut.write('  );\n\n')
  lut.write('  always_comb begin\n')
  lut.write('    case (d)\n')

  # Sum can be from 0 to 400 inclusive
  for i in range(256):
    init = format(i, 'b').zfill(8)
    res = format(int(128 * abs(math.tanh(i/32))), 'b').zfill(8)
    lut.write('      8\'b{0}: q <= 8\'b{1};\n'.format(init, res))

  lut.write('      default: q <= 0;\n')
  lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('endmodule\n')
