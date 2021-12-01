import numpy as np
import struct
import getopt
import math
import sys

def read_neuron(file, num_inputs):
  bias = struct.unpack('f', file.read(4))[0]
  weights = list(struct.unpack('f' * num_inputs, file.read(4 * num_inputs)))
  return (bias, weights)

def pack_weight(weight):
  return struct.pack('f' * (len(weight[1]) + 1), weight[0], *weight[1])

def gen_new_weight(fan_in, fan_out):
  dist = math.sqrt(6 / (fan_in + fan_out))
  weights = np.random.uniform(low=-dist, high=dist, size=(fan_in,))
  return (0, weights)

def pack_new_weight(fan_in, fan_out):
  return pack_weight(gen_new_weight(fan_in, fan_out))

def gen_new_dnn_weights(path):
  with open(path, 'wb+') as file:
    for _ in range(4):
      file.write(pack_new_weight(100, 1))
    for _ in range(16):
      file.write(pack_new_weight(25, 1))
    for _ in range(5):
      file.write(pack_new_weight(80, 1))
    file.write(pack_new_weight(4, 1))
    file.write(pack_new_weight(16, 1))
    file.write(pack_new_weight(5, 1))
    file.write(pack_new_weight(3, 1))

def read_dnn_weights(path):
  dnn = []
  with open(path, 'rb') as file:
    a_t1_weights = []
    a_t2_weights = []
    a_t3_weights = []
    b_weights = []
    for _ in range(4):
      a_t1_weights.append(read_neuron(file, 100))
    for _ in range(16):
      a_t2_weights.append(read_neuron(file, 25))
    for _ in range(5):
      a_t3_weights.append(read_neuron(file, 80))
    b_weights.append(read_neuron(file, 4))
    b_weights.append(read_neuron(file, 16))
    b_weights.append(read_neuron(file, 5))
    dnn.append(a_t1_weights)
    dnn.append(a_t2_weights)
    dnn.append(a_t3_weights)
    dnn.append(b_weights)
    dnn.append(read_neuron(file, 3))
  return dnn

def gen_new_rnn_weights(path):
  with open(path, 'wb+') as file:
    for _ in range(15):
      file.write(pack_new_weight(400, 15))
    for _ in range(15):
      file.write(pack_new_weight(15, 15))
    for _ in range(36):
      file.write(pack_new_weight(15, 1))

def read_rnn_weights(path):
  rnn = []
  with open(path, 'rb') as file:
    a_weights = []
    b_weights = []
    c_weights = []
    for _ in range(15):
      a_weights.append(read_neuron(file, 400))
    for _ in range(15):
      b_weights.append(read_neuron(file, 15))
    for _ in range(36):
      c_weights.append(read_neuron(file, 15))
    rnn.append(a_weights)
    rnn.append(b_weights)
    rnn.append(c_weights)
  return rnn

def write_rnn_weights(path, weights):
  with open(path, 'wb+') as file:
    for i in range(15):
      file.write(pack_weight(weights[0][i]))
    for i in range(15):
      file.write(pack_weight(weights[1][i]))
    for i in range(36):
      file.write(pack_weight(weights[2][i]))

def write_dnn_weights(path, weights):
  with open(path, 'wb+') as file:
    for i in range(4):
      file.write(weights[0][i])
    for i in range(16):
      file.write(weights[1][i])
    for i in range(5):
      file.write(weights[2][i])
    for i in range(3):
      file.write(weights[3][i])
    file.write(weights[4])

if __name__ == '__main__':
  sys_args = sys.argv[1:]
  output_path = None
  mode = None
  try:
    args, vals = getopt.getopt(sys_args, 'o:m:', ['out =', 'mode ='])
    for arg, val in args:
      if arg in ('-o', '--out'):
        output_path = val
      elif arg in ('-m', '--mode'):
        mode = val
  except getopt.error as err:
    print(str(err))
    exit()
  if not mode or mode not in ('gen-rnw', 'gen-dnw'):
    print('Invalid mode')
    exit()
  if mode == 'gen-rnw':
    gen_new_rnn_weights(output_path if output_path else './rnw.bin')
  elif mode == 'gen-dnw':
    gen_new_dnn_weights(output_path if output_path else './dnw.bin')
