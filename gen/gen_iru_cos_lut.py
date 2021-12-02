import math

with open('../hw/iru/iru_cos_lut.sv', 'w+') as lut:
  lut.write('module iru_cos_lut\n')
  lut.write('  (\n')
  lut.write('    /*\n')
  lut.write('     * Inputs\n')
  lut.write('     */\n')
  lut.write('    input [5:0] d,\n\n')
  lut.write('    /*\n')
  lut.write('     * Outputs\n')
  lut.write('     */\n')
  lut.write('    output reg [8:0] q\n')
  lut.write('  );\n\n')
  lut.write('  always_comb begin\n')
  lut.write('    case (d)\n')

  for i in range(36):
    init = format(i, 'b').zfill(6)
    value = math.cos(math.radians(i * 10))
    sign = '1' if value < 0 else '0'
    res = format(int(128 * abs(value)), 'b').zfill(8)

    lut.write('      9\'b{0}: q <= 8\'b{1}{2};\n'.format(init, sign, res))

  lut.write('      default: q <= 0;\n')
  lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('endmodule\n')
