from re import L
from neuron import Neuron

import math

DNN_GAMMA = 0.05
DNN_BIAS_GAMMA = 0.01

def dtanh(value):
  return 1 - math.tanh(value)**2

class DNN:

  def __init__(self, data):

    self.a_layer_type1 = []
    self.a_layer_type2 = []
    self.a_layer_type3 = []

    for i in range(2):
      row = []
      for j in range(2):
        row.append(Neuron(data['a_layer']['type1'][i][j]))
      self.a_layer_type1.append(row)
    for i in range(4):
      row = []
      for j in range(4):
        row.append(Neuron(data['a_layer']['type2'][i][j]))
      self.a_layer_type2.append(row)
    for i in range(5):
      self.a_layer_type3.append(Neuron(data['a_layer']['type3'][i]))

    self.b_type1 = Neuron(data['b_type1'])
    self.b_type2 = Neuron(data['b_type2'])
    self.b_type3 = Neuron(data['b_type3'])

    self.c = Neuron(data['c'])

  def arrange_input(self, input):
    type1_in = []
    for i in range(2):
      row = []
      for j in range(2):
        row.append([])
      type1_in.append(row)

    for i in range(20):
      for j in range(20):
        b_i = i // 10
        b_j = j // 10
        type1_in[b_i][b_j].append(input[(i * 20) + j])

    type2_in = []
    for i in range(4):
      row = []
      for j in range(4):
        row.append([])
      type2_in.append(row)
    
    for i in range(20):
      for j in range(20):
        b_i = i // 5
        b_j = j // 5
        type2_in[b_i][b_j].append(input[(i * 20) + j])

    type3_in = []
    for i in range(5):
      type3_in.append([])

    for i in range(20):
      for j in range(20):
        b_i = i // 4
        type3_in[b_i].append(input[(i * 20) + j])
    return (type1_in, type2_in, type3_in)

  def train(self, input, expected_output):
  
    self.forward(input)

    type1_in, type2_in, type3_in = self.arrange_input(input)

    c_err = (self.c.activation - expected_output) * dtanh(self.c.activation)

    b_type1_err = c_err * self.c.weights[0] * dtanh(self.b_type1.activation)
    b_type2_err = c_err * self.c.weights[1] * dtanh(self.b_type2.activation)
    b_type3_err = c_err * self.c.weights[2] * dtanh(self.b_type3.activation)

    type1_err = []
    type2_err = []
    type3_err = []

    # Calc err for a layer type 1
    for i in range(2):
      row = []
      for j in range(2):
        row.append(b_type1_err * self.b_type1.weights[(i * 2) + j] * dtanh(self.a_layer_type1[i][j].activation))
      type1_err.append(row)

    # Calc err for a layer type 2
    for i in range(4):
      row = []
      for j in range(4):
        row.append(b_type2_err * self.b_type2.weights[(i * 4) + j] * dtanh(self.a_layer_type2[i][j].activation))
      type2_err.append(row)

    # Calc err for a layer type 3
    for i in range(5):
      type3_err.append(b_type3_err * self.b_type3.weights[i] * dtanh(self.a_layer_type3[i].activation))
    
    # Update a layer type 1
    for i in range(2):
      for j in range(2):
        for k in range(100):
          self.a_layer_type1[i][j].weights[k] -= DNN_GAMMA * type1_err[i][j] * type1_in[i][j][k]
        self.a_layer_type1[i][j].bias -= DNN_BIAS_GAMMA * type1_err[i][j]
    
    # Update a layer type 2
    for i in range(4):
      for j in range(4):
        for k in range(25):
          self.a_layer_type2[i][j].weights[k] -= DNN_GAMMA * type2_err[i][j] * type2_in[i][j][k]
        self.a_layer_type2[i][j].bias -= DNN_BIAS_GAMMA * type2_err[i][j]

    # Update a layer type 3
    for i in range(5):
      for j in range(80):
        self.a_layer_type3[i].weights[k] -= DNN_GAMMA * type3_err[i] * type3_in[i][k]
      self.a_layer_type3[i].bias -= DNN_BIAS_GAMMA * type3_err[i]

    # Update b type1
    for i in range(2):
      for j in range(2):
        self.b_type1.weights[(i * 2) + j] -= DNN_GAMMA * b_type1_err * self.a_layer_type1[i][j].activation
    self.b_type1.bias -= DNN_BIAS_GAMMA * b_type1_err

    # Update b type2
    for i in range(4):
      for j in range(4):
        self.b_type2.weights[(i * 4) + j] -= DNN_GAMMA * b_type2_err * self.a_layer_type2[i][j].activation
    self.b_type2.bias -= DNN_BIAS_GAMMA * b_type2_err

    # Update b type3
    for i in range(5):
      self.b_type3.weights[i] -= DNN_GAMMA * b_type3_err * self.a_layer_type3[i].activation
    self.b_type3.bias -= DNN_BIAS_GAMMA * b_type3_err

    # Update c
    self.c.weights[0] -= DNN_GAMMA * c_err * self.b_type1.activation
    self.c.weights[1] -= DNN_GAMMA * c_err * self.b_type2.activation
    self.c.weights[2] -= DNN_GAMMA * c_err * self.b_type3.activation
    
    self.c.bias -= DNN_BIAS_GAMMA * c_err

  def forward(self, input):

    type1_in, type2_in, type3_in = self.arrange_input(input)

    # A layer
    type1_out = []
    type2_out = []
    type3_out = []

    for i in range(2):
      for j in range(2):
        self.a_layer_type1[i][j].forward(type1_in[i][j])
        type1_out.append(self.a_layer_type1[i][j].activation)

    for i in range(4):
      for j in range(4):
        self.a_layer_type2[i][j].forward(type2_in[i][j])
        type2_out.append(self.a_layer_type2[i][j].activation)

    for i in range(5):
      self.a_layer_type3[i].forward(type3_in[i])
      type3_out.append(self.a_layer_type3[i].activation)

    self.b_type1.forward(type1_out)
    self.b_type2.forward(type2_out)
    self.b_type3.forward(type3_out)

    b_out = [
      self.b_type1.activation,
      self.b_type2.activation,
      self.b_type3.activation
    ]

    self.c.forward(b_out)

    return 1 if self.c.activation > 0 else 0
