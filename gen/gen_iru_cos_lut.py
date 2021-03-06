from bitstring import Bits

import math

NUM_BITS = 32

with open('../hw/iru/iru_cos_lut.sv', 'w+') as lut:
  lut.write('module iru_cos_lut\n')
  lut.write('  (\n')
  lut.write('    /*\n')
  lut.write('     * Inputs\n')
  lut.write('     */\n')
  lut.write('    input [35:0] d,\n\n')
  lut.write('    /*\n')
  lut.write('     * Outputs\n')
  lut.write('     */\n')
  lut.write('    output reg [{0}:0] q\n'.format(NUM_BITS - 1))
  lut.write('  );\n\n')
  lut.write('  always_comb begin\n')
  lut.write('    case (d)\n')

  for i in range(36):
    init = ['0' for _ in range(36)]
    init[i] = '1'
    value = math.cos(math.radians(i * 10))
    res = Bits(int=int(2**(NUM_BITS - 2) * value), length=NUM_BITS)
    lut.write('      36\'b{0}: q <= {1}\'b{2};\n'.format(''.join(init)[::-1], NUM_BITS, res.bin))

  lut.write('      default: q <= \'0;\n')
  lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('endmodule\n')
