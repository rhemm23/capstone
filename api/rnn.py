from neuron import Neuron

import math

class RNN:
  def __init__(self, data):
    self.a_layer = []
    self.b_layer = []
    self.c_layer = []
    for i in range(15):
      self.a_layer.append(Neuron())
    for i in range(15):
      self.b_layer.append(Neuron(data['b_layer'][i]))
    for i in range(36):
      self.c_layer.append(Neuron(data['c_layer'][i]))

  def forward(self, input):

    a_out = []
    b_out = []
    c_out = []

    # Feed forward
    for i in range(15):
      a_out.append(self.a_layer[i].forward(input))
    for i in range(15):
      b_out.append(self.b_layer[i].forward(a_out))
    for i in range(36):
      c_out.append(self.c_layer[i].forward(b_out))
    
    # Softmax
    e_sum = 0
    for i in range(36):
      e_sum += math.exp(c_out[i])
    
    c_prob = []
    for i in range(36):
      c_prob.append(math.exp(c_out[i]) / e_sum)

    return c_prob
