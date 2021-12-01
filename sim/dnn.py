from math import tanh

def arrange_input(input):
  t1_in = [[] for _ in range(4)]
  t2_in = [[] for _ in range(16)]
  t3_in = [[] for _ in range(5)]
  for i in range(20):
    for j in range(20):
      b_i = i // 10
      b_j = j // 10
      t1_in[(b_i * 2) + b_j].append(input[(i * 20) + j])
  for i in range(20):
    for j in range(20):
      b_i = i // 5
      b_j = j // 5
      t2_in[(b_i * 4) + b_j].append(input[(i * 20) + j])
  for i in range(20):
    for j in range(20):
      b_i = i // 4
      t3_in[b_i].append(input[(i * 20) + j])
  return (t1_in, t2_in, t3_in)

def dnn_layer_out(sub_image, weights):
  a1_out = []
  a2_out = []
  a3_out = []
  a1_in, a2_in, a3_in = arrange_input(sub_image)
  for i in range(4):
    act = weights[0][(i * 2) + j][0]
    for j in range(100):
      act += a1_in[i][j] * weights[0][(i * 2) + j][1][j]
    a1_out.append(tanh(act))
  for i in range(16):
    act = weights[1][(i * 4) + j][0]
    for j in range(25):
      act += a2_in[i][j] * weights[1][(i * 4) + j][1][j]
    a2_out.append(tanh(act))
  for i in range(5):
    act = weights[2][i][0]
    for j in range(80):
      act += a3_in[i][j] * weights[2][i][1][j]
    a3_out.append(tanh(act))
  b1_out = weights[3][0][0]
  b2_out = weights[3][1][0]
  b3_out = weights[3][2][0]
  for i in range(4):
    b1_out += a1_out[i] * weights[3][0][1][i]
  for i in range(16):
    b2_out += a2_out[i] * weights[3][1][1][i]
  for i in range(5):
    b3_out += a3_out[i] * weights[3][2][1][i]
  b1_out = tanh(b1_out)
  b2_out = tanh(b2_out)
  b3_out = tanh(b3_out)
  c_out = weights[4][0]
  c_out += weights[4][1][0] * b1_out
  c_out += weights[4][1][1] * b2_out
  c_out += weights[4][1][2] * b3_out
  c_out = tanh(c_out)
  return ([a1_out, a2_out, a3_out], [b1_out, b2_out, b3_out], c_out)

def dnn(sub_image, weights):
  _, _, c_out = dnn_layer_out(sub_image, weights)
  return 1 if c_out > 0 else 0
