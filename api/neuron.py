import math

class Neuron:
  def __init__(self, data):
    self.bias = data['bias']
    self.weights = data['weights']

  def forward(self, inputs):
    act = self.bias
    for i in range(len(inputs)):
      act += inputs[i] * self.weights[i]
    return math.tanh(act)