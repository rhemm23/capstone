from bitstring import Bits, BitArray

import torch

class BinModelLoader:
  def read_float(self):
    value = Bits(uint=self.bytes[self.byte_cnt], length=8).int
    self.byte_cnt += 1
    return value / 256

  def save(self, state_dict, path):
    layers = []
    for name in state_dict:
      layer = name.split('.')[0]
      if layer not in layers:
        layers.append(layer)
    binary = BitArray()
    for layer in layers:
      bias = state_dict[layer + '.bias'].tolist()
      weights = state_dict[layer + '.weight'].tolist()
      for i in range(len(bias)):
        bias_f = int(bias[i] * 256)
        binary.append(Bits(int=bias_f, length=8))
        for weight in weights[i]:
          weight_f = int(weight * 256)
          binary.append(Bits(int=weight_f, length=8))
    with open(path, 'wb+') as file:
      file.write(binary.bytes)

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
      


