from math import tanh

def rnn_layer_out(sub_image, weights):
  a_out = []
  b_out = []
  c_out = []
  for i in range(15):
    act = weights[0][i][0]
    for j in range(400):
      act += sub_image[j] * weights[0][i][1][j]
    a_out.append(tanh(act))
  for i in range(15):
    act = weights[1][i][0]
    for j in range(15):
      act += a_out[j] * weights[1][i][1][j]
    b_out.append(tanh(act))
  for i in range(36):
    act = weights[2][i][0]
    for j in range(15):
      act += b_out[j] * weights[1][i][1][j]
    c_out.append(tanh(act))
  return (a_out, b_out, c_out)

def rnn(sub_image, weights):
  _, _, c_out = rnn_layer_out(sub_image, weights)
  return [1 if c_out[i] > 0 else 0 for i in range(36)]
