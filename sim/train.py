from rnn import *
from dnn import *

import math

RNN_GAMMA = 0.05
RNN_BIAS_GAMMA = 0.01

DNN_GAMMA = 0.05
DNN_BIAS_GAMMA = 0.01

def dtanh(value):
  return 1 - math.tanh(value)**2

def train_rnn(sub_image, expected_output, rnw):
  a_out, b_out, c_out = rnn_layer_out(sub_image, rnw)
  c_err = [(c_out[i] - expected_output[i]) * dtanh(c_out[i]) for i in range(36)]
  b_err = []
  a_err = []
  for i in range(15):
    err = 0
    for j in range(36):
      err += c_err[j] * rnw[2][j][1][i]
    err *= dtanh(b_out[i])
    b_err.append(err)
  for i in range(15):
    err = 0
    for j in range(15):
      err += b_err[j] * rnw[1][j][1][i]
    err *= dtanh(a_out[i])
    a_err.append(err)
  for i in range(15):
    for j in range(400):
      rnw[0][i][1][j] -= RNN_GAMMA * a_err[i] * sub_image[j]
    rnw[0][i] = (rnw[0][i][0] - RNN_BIAS_GAMMA * a_err[i], rnw[0][i][1])
  for i in range(15):
    for j in range(15):
      rnw[1][i][1][j] -= RNN_GAMMA * b_err[i] * a_out[j]
    rnw[1][i] = (rnw[1][i][0] - RNN_BIAS_GAMMA * b_err[i], rnw[1][i][1])
  for i in range(36):
    for j in range(15):
      rnw[2][i][1][j] -= RNN_GAMMA * c_err[i] * b_out[j]
    rnw[2][i] = (rnw[2][i][0] - RNN_BIAS_GAMMA * c_err[i], rnw[2][i][1])

def train_dnn(sub_image, expected_output, dnw):
  a1_in, a2_in, a3_in = arrange_input(sub_image)
  a_out, b_out, c_out = dnn_layer_out(sub_image, dnw)
  c_err = (c_out - expected_output) * dtanh(c_out)
  b_err = [c_err * dnw[4][1][i] * dtanh(b_out[i]) for i in range(3)]
  a1_err = [b_err[0] * dnw[3][0][1][i] * dtanh(a_out[0][i]) for i in range(4)]
  a2_err = [b_err[1] * dnw[3][1][1][i] * dtanh(a_out[1][i]) for i in range(16)]
  a3_err = [b_err[2] * dnw[3][2][1][i] * dtanh(a_out[2][i]) for i in range(5)]
  for i in range(4):
    for j in range(100):
      dnw[0][i][1][j] -= DNN_GAMMA * a1_err[i] * a1_in[i][j]
    dnw[0][i] = (dnw[0][i][0] - DNN_BIAS_GAMMA * a1_err[i], dnw[0][i][1])
  for i in range(16):
    for j in range(25):
      dnw[1][i][1][j] -= DNN_GAMMA * a2_err[i] * a2_in[i][j]
    dnw[1][i] = (dnw[1][i][0] - DNN_BIAS_GAMMA * a2_err[i], dnw[1][i][1])
  for i in range(5):
    for j in range(80):
      dnw[2][i][1][j] -= DNN_GAMMA * a3_err[i] * a3_in[i][j]
    dnw[2][i] = (dnw[2][i][0] - DNN_BIAS_GAMMA * a3_err[i], dnw[2][i][1])
  for i in range(4):
    dnw[3][0][1][i] -= DNN_GAMMA * b_err[0] * a_out[0][i]
  dnw[3][0] = (dnw[3][0][0] - DNN_BIAS_GAMMA * b_err[0], dnw[3][0][1])
  for i in range(16):
    dnw[3][1][1][i] -= DNN_GAMMA * b_err[1] * a_out[1][i]
  dnw[3][1] = (dnw[3][1][0] - DNN_BIAS_GAMMA * b_err[1], dnw[3][1][1])
  for i in range(5):
    dnw[3][2][1][i] -= DNN_GAMMA * b_err[2] * a_out[2][i]
  dnw[3][2] = (dnw[3][2][0] - DNN_BIAS_GAMMA * b_err[2], dnw[3][2][1])
  for i in range(3):
    dnw[4][1][i] -= DNN_GAMMA * c_err * b_out[i]
  dnw[4] = (dnw[4][0] - DNN_BIAS_GAMMA * c_err, dnw[4][1])
