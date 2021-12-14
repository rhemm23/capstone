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

# A layer
for i in range(15):

  # Neuron data
  bin_data += float_to_bytes(params['lin1.bias'][i])
  for j in range(400):
    bin_data += float_to_bytes(params['lin1.weight'][i][j])

  # Page align
  for j in range(7):
    bin_data += float_to_bytes(0)

# B layer
for i in range(30):

  # Neuron data
  bin_data += float_to_bytes(params['lin2.bias'][i])
  for j in range(15):
    bin_data += float_to_bytes(params['lin2.weight'][i][j])

# C layer
for i in range(36):

  # Neuron data
  bin_data += float_to_bytes(params['lin3.bias'][i])
  for j in range(30):
    bin_data += float_to_bytes(params['lin3.weight'][i][j])

  # Page align
  bin_data += float_to_bytes(0)

with open('./rnw.bin', 'wb+') as out:
  out.write(bin_data)
