# Open heu sum lut verilog file
with open('../hw/ipgu/ipgu_mult_lut.sv', 'w+') as lut:
  lut.write('module ipgu_mult_lut\n')
  lut.write('  (\n')
  lut.write('    /*\n')
  lut.write('     * Inputs\n')
  lut.write('     */\n')
  lut.write('    input [2:0] scaleNum,//which downscale ratio to use/multiplier\n')
  lut.write('    input [8:0] addr,//9 bits for addr\n\n')
  lut.write('    /*\n')
  lut.write('     * Outputs\n')
  lut.write('     */\n')
  lut.write('    output wire [8:0] addrOut\n')
  lut.write('  );\n\n')
  ratios = [0.8, 0.75, 2/3, 0.5, 1/3]
  numAddr = [300, 240, 180, 120, 60, 20]
  lut.write('  reg [8:0] q [5];\n')
  lut.write('  always_comb begin\n')
  for i in range(len(ratios)):
      ratio = ratios[i]
      numBits = numAddr[i].bit_length()
      lut.write('    case (addr['+ str(numBits)+'-1:0])\n')

      # Sum can be from 0 to 400 inclusive
      for j in range(numAddr[i]):

          init = format(j, 'b').zfill(numBits)
          res = format(int((j * ratio)), 'b').zfill(9)
          lut.write(('      '+str(numBits)+'\'b'+init+': q['+str(i)+'] <= 9\'b'+res+';\n'))

      lut.write('      default: q['+str(i)+'] <= \'0;\n')
      lut.write('    endcase\n')
  lut.write('  end\n\n')
  lut.write('  assign addrOut = scaleNum<5?q[scaleNum]:\'1\n')
  lut.write('endmodule\n')
