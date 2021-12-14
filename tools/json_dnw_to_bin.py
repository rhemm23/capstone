import bitstring
import json
import sys

if len(sys.argv) != 2:
  print('Expected path to json')
  exit()

params = None
with open(sys.argv[1], 'r') as file:
  params = json.load(file)

def float_to_bytes(value):
  return bitstring.Bits(float=value, length=64).bytes

bin_data = bytes()

for i in range(4):
  bin_data += float_to_bytes(params['a1_{}.bias'.format(i)][0])
  for j in range(100):
    bin_data += float_to_bytes(params['a1_{}.weight'.format(i)][0][j])
  for j in range(3):
    bin_data += float_to_bytes(0)

for i in range(16):
  bin_data += float_to_bytes(params['a2_{}.bias'.format(i)][0])
  for j in range(25):
    bin_data += float_to_bytes(params['a2_{}.weight'.format(i)][0][j])
  for j in range(6):
    bin_data += float_to_bytes(0)

for i in range(5):
  bin_data += float_to_bytes(params['a3_{}.bias'.format(i)][0])
  for j in range(80):
    bin_data += float_to_bytes(params['a3_{}.weight'.format(i)][0][j])
  for j in range(7):
    bin_data += float_to_bytes(0)

bin_data += float_to_bytes(params['b1.bias'][0])
for i in range(4):
  bin_data += float_to_bytes(params['b1.weight'][0][i])
for i in range(3):
  bin_data += float_to_bytes(0)

bin_data += float_to_bytes(params['b2.bias'][0])
for i in range(16):
  bin_data += float_to_bytes(params['b2.weight'][0][i])
for i in range(7):
  bin_data += float_to_bytes(0)

bin_data += float_to_bytes(params['b3.bias'][0])
for i in range(5):
  bin_data += float_to_bytes(params['b3.weight'][0][i])
for i in range(2):
  bin_data += float_to_bytes(0)

bin_data += float_to_bytes(params['c.bias'][0])
for i in range(3):
  bin_data += float_to_bytes(params['c.weight'][0][i])
for i in range(4):
  bin_data += float_to_bytes(0)

with open('./dnw.bin', 'wb+') as file:
  file.write(bin_data)
