import math

class Neuron:
  def __init__(self, data):
    self.bias = data['bias']
    self.weights = data['weights']
    self.activation = 0

  def forward(self, inputs):
    act = self.bias
    for i in range(len(inputs)):
      act += inputs[i] * self.weights[i]
    self.activation = math.tanh(act)

  def dict(self):
    return {
      'bias': self.bias,
      'weights': self.weights
    }