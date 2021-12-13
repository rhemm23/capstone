from bitstring import Bits

import math

CONSTRAST = 128

# Open tanh lut verilog file
with open('../hw/bcau/bcau_lut.sv', 'w+') as lut:
  lut.write('module bcau_lut\n')
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

  factor = (259 * (255 + CONSTRAST)) / (255 * (259 - CONSTRAST))

  for i in range(256):
    init = Bits(uint=i, length=8)
    temp = int(factor * (i - CONSTRAST) + CONSTRAST)
    temp = max(0, min(255, temp))
    temp = Bits(uint=temp, length=8)
    lut.write('      8\'b{0}: q <= 8\'b{1};\n'.format(init.bin, temp.bin))

  lut.write('      default: q <= \'0;\n')
  lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('endmodule\n')
