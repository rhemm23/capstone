from neuron import Neuron

import math

RNN_GAMMA = 0.05
RNN_BIAS_GAMMA = 0.01

def dtanh(value):
  return 1 - math.tanh(value)**2

class RNN:
  def __init__(self, data):
    self.a_layer = []
    self.b_layer = []
    self.c_layer = []
    for i in range(15):
      self.a_layer.append(Neuron(data['a_layer'][i]))
    for i in range(15):
      self.b_layer.append(Neuron(data['b_layer'][i]))
    for i in range(36):
      self.c_layer.append(Neuron(data['c_layer'][i]))

  def dict(self):
    return {
      'a_layer': [self.a_layer[i].dict() for i in range(15)],
      'b_layer': [self.b_layer[i].dict() for i in range(15)],
      'c_layer': [self.c_layer[i].dict() for i in range(36)]
    }

  def train(self, input, expected_output):
    self.forward(input)
    c_err = []
    b_err = []
    a_err = []
    for i in range(36):
      c_out = self.c_layer[i].activation
      c_err.append((c_out - expected_output[i]) * dtanh(c_out))
    for i in range(15):
      err = 0
      for j in range(36):
        err += c_err[j] * self.c_layer[j].weights[i]
      err *= dtanh(self.b_layer[i].activation)
      b_err.append(err)
    for i in range(15):
      err = 0
      for j in range(15):
        err += b_err[j] * self.b_layer[j].weights[i]
      err *= dtanh(self.a_layer[i].activation)
      a_err.append(err)

    # Update a neuron weights
    for i in range(15):
      for j in range(400):
        self.a_layer[i].weights[j] -= RNN_GAMMA * a_err[i] * input[j]
      self.a_layer[i].bias -= RNN_BIAS_GAMMA * a_err[i]

    # Update b neuron weights
    for i in range(15):
      for j in range(15):
        self.b_layer[i].weights[j] -= RNN_GAMMA * b_err[i] * self.a_layer[j].activation
      self.b_layer[i].bias -= RNN_BIAS_GAMMA * b_err[i]
  
    # Update c neuron weights
    for i in range(36):
      for j in range(15):
        self.c_layer[i].weights[j] -= RNN_GAMMA * c_err[i] * self.b_layer[j].activation
      self.c_layer[i].bias -= RNN_BIAS_GAMMA * c_err[i]

  def forward(self, input):
    a_out = []
    b_out = []
    c_out = []
    for i in range(15):
      self.a_layer[i].forward(input)
      a_out.append(self.a_layer[i].activation)
    for i in range(15):
      self.b_layer[i].forward(a_out)
      b_out.append(self.b_layer[i].activation)
    for i in range(36):
      self.c_layer[i].forward(b_out)
      bit_res = 1 if self.c_layer[i].activation > 0 else 0
      c_out.append(bit_res)
    return c_out
