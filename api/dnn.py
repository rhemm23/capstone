from re import L
from neuron import Neuron

import numpy as np

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

  def forward(self, input):

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
