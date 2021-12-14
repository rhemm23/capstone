from torch import nn

import torch

class RotNet(nn.Module):
  def __init__(self):
    super(RotNet, self).__init__()

    self.lin1 = nn.Linear(400, 15)
    self.lin2 = nn.Linear(15, 30)
    self.lin3 = nn.Linear(30, 36)

    self.act = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)

  def get_prediction(self, x):
    x1 = torch.empty((1, 400), dtype=torch.double)
    x1[0] = x
    res = self.forward(x1.float()).tolist()
    index = res[0].index(max(res[0]))
    rnn_out = [False for _ in range(36)]
    rnn_out[index] = True
    return rnn_out

  def forward(self, x):
    x = self.act(self.lin1(x))
    x = self.act(self.lin2(x))
    return self.softmax(self.lin3(x))

class DetNet(nn.Module):
  def __init__(self):
    super(DetNet, self).__init__()

    self.a1_0 = nn.Linear(100, 1)
    self.a1_1 = nn.Linear(100, 1)
    self.a1_2 = nn.Linear(100, 1)
    self.a1_3 = nn.Linear(100, 1)

    self.a2_0 = nn.Linear(25, 1)
    self.a2_1 = nn.Linear(25, 1)
    self.a2_2 = nn.Linear(25, 1)
    self.a2_3 = nn.Linear(25, 1)
    self.a2_4 = nn.Linear(25, 1)
    self.a2_5 = nn.Linear(25, 1)
    self.a2_6 = nn.Linear(25, 1)
    self.a2_7 = nn.Linear(25, 1)
    self.a2_8 = nn.Linear(25, 1)
    self.a2_9 = nn.Linear(25, 1)
    self.a2_10 = nn.Linear(25, 1)
    self.a2_11 = nn.Linear(25, 1)
    self.a2_12 = nn.Linear(25, 1)
    self.a2_13 = nn.Linear(25, 1)
    self.a2_14 = nn.Linear(25, 1)
    self.a2_15 = nn.Linear(25, 1)

    self.a3_0 = nn.Linear(80, 1)
    self.a3_1 = nn.Linear(80, 1)
    self.a3_2 = nn.Linear(80, 1)
    self.a3_3 = nn.Linear(80, 1)
    self.a3_4 = nn.Linear(80, 1)

    self.b1 = nn.Linear(4, 1)
    self.b2 = nn.Linear(16, 1)
    self.b3 = nn.Linear(5, 1)

    self.c = nn.Linear(3, 1)

    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

  def arrange_input(self, input):
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

  def forward(self, x):

    a1_in = [torch.empty(len(x), 100) for _ in range(4)]
    a2_in = [torch.empty(len(x), 25) for _ in range(16)]
    a3_in = [torch.empty(len(x), 80) for _ in range(5)]

    a1_out, a2_out, a3_out = [], [], []

    # Iterate over batches
    for i in range(len(x)):
      a1_tin, a2_tin, a3_tin = self.arrange_input(x[i].tolist())
      for j in range(4):
        a1_in[j][i] = torch.tensor(a1_tin[j])
      for j in range(16):
        a2_in[j][i] = torch.tensor(a2_tin[j])
      for j in range(5):
        a3_in[j][i] = torch.tensor(a3_tin[j])

    a1_out.append(self.tanh(self.a1_0(a1_in[0])))
    a1_out.append(self.tanh(self.a1_1(a1_in[1])))
    a1_out.append(self.tanh(self.a1_2(a1_in[2])))
    a1_out.append(self.tanh(self.a1_3(a1_in[3])))

    a2_out.append(self.tanh(self.a2_0(a2_in[0])))
    a2_out.append(self.tanh(self.a2_1(a2_in[1])))
    a2_out.append(self.tanh(self.a2_2(a2_in[2])))
    a2_out.append(self.tanh(self.a2_3(a2_in[3])))
    a2_out.append(self.tanh(self.a2_4(a2_in[4])))
    a2_out.append(self.tanh(self.a2_5(a2_in[5])))
    a2_out.append(self.tanh(self.a2_6(a2_in[6])))
    a2_out.append(self.tanh(self.a2_7(a2_in[7])))
    a2_out.append(self.tanh(self.a2_8(a2_in[8])))
    a2_out.append(self.tanh(self.a2_9(a2_in[9])))
    a2_out.append(self.tanh(self.a2_10(a2_in[10])))
    a2_out.append(self.tanh(self.a2_11(a2_in[11])))
    a2_out.append(self.tanh(self.a2_12(a2_in[12])))
    a2_out.append(self.tanh(self.a2_13(a2_in[13])))
    a2_out.append(self.tanh(self.a2_14(a2_in[14])))
    a2_out.append(self.tanh(self.a2_15(a2_in[15])))

    a3_out.append(self.tanh(self.a3_0(a3_in[0])))
    a3_out.append(self.tanh(self.a3_1(a3_in[1])))
    a3_out.append(self.tanh(self.a3_2(a3_in[2])))
    a3_out.append(self.tanh(self.a3_3(a3_in[3])))
    a3_out.append(self.tanh(self.a3_4(a3_in[4])))

    c_in = torch.cat([
      self.tanh(self.b1(torch.cat(a1_out, dim=1))),
      self.tanh(self.b2(torch.cat(a2_out, dim=1))),
      self.tanh(self.b3(torch.cat(a3_out, dim=1)))
    ], dim=1)

    return self.sigmoid(self.c(c_in))

