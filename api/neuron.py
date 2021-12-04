import math

class Neuron:
  def __init__(self, bias, weights):
    self.weights = weights
    self.bias = bias

  def forward(self, inputs):
    act = self.bias
    for i in range(len(inputs)):
      act += inputs[i] * self.weights[i]
    return math.tanh(act)