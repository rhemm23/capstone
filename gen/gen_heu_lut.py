# Open heu sum lut verilog file
with open('../hw/heu/heu_sum_lut.sv', 'w+') as lut:
  lut.write('module heu_sum_lut\n')
  lut.write('  (\n')
  lut.write('    /*\n')
  lut.write('     * Inputs\n')
  lut.write('     */\n')
  lut.write('    input [8:0] d,\n\n')
  lut.write('    /*\n')
  lut.write('     * Outputs\n')
  lut.write('     */\n')
  lut.write('    output reg [7:0] q\n')
  lut.write('  );\n\n')
  lut.write('  always_comb begin\n')
  lut.write('    case (d)\n')

  # Sum can be from 0 to 400 inclusive
  for i in range(401):

    init = format(i, 'b').zfill(9)
    res = format(int((i / 400) * 255), 'b').zfill(8)

    lut.write('      9\'b{0}: q <= 8\'b{1};\n'.format(init, res))

  lut.write('      default: q <= 0;\n')
  lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('endmodule\n')
