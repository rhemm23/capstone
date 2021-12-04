from bitstring import Bits, BitArray, CreationError

import torch
import math

class BinModelLoader:
  def __init__(self, bit_cnt):
    self.bit_cnt = bit_cnt
    self.byte_size = math.ceil(bit_cnt / 8)
    self.factor = 2**(bit_cnt - 4)

  def read_float(self):
    value = Bits(bytes=self.bytes[self.byte_cnt:self.byte_cnt + self.byte_size], length=self.bit_cnt).int
    self.byte_cnt += self.byte_size
    return value / self.factor

  def to_float(self, value):
    try:
      scaled = int(value * self.factor)
      return Bits(int=scaled, length=self.bit_cnt).bytes
    except CreationError:
      print('Invalid value: {}'.format(value))
      exit()

  def save(self, path, model):
    layers = []
    state_dict = model.state_dict()
    for name in state_dict:
      layer = name.split('.')[0]
      if layer not in layers:
        layers.append(layer)
    data = bytes()
    for layer in layers:
      bias = state_dict[layer + '.bias'].tolist()
      weights = state_dict[layer + '.weight'].tolist()
      for i in range(len(bias)):
        data += self.to_float(bias[i])
        for weight in weights[i]:
          data += self.to_float(weight)
    with open(path, 'wb+') as file:
      file.write(data)

  def load(self, path, model, device):
    with open(path, 'rb') as file:
      self.bytes = file.read()
      self.byte_cnt = 0
      layers = []
      for name, weights in model.named_parameters():
        parts = name.split('.')
        if parts[1] == 'weight' and parts[0] not in layers:
          shape = weights.shape
          layers.append((parts[0], shape[0], shape[1]))
      state = {}
      for layer, out_size, in_size in layers:
        biases = []
        weights = []
        for i in range(out_size):
          biases.append(self.read_float())
          neuron_weights = []
          for j in range(in_size):
            neuron_weights.append(self.read_float())
          weights.append(neuron_weights)
        state[layer + '.bias'] = torch.tensor(biases).to(device=device)
        state[layer + '.weight'] = torch.tensor(weights).to(device=device)
      model.load_state_dict(state)
      


