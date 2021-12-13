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

  def forward(self, x):
    x = self.act(self.lin1(x))
    x = self.act(self.lin2(x))
    return self.softmax(self.lin3(x))

class DetNet(nn.Module):
  def __init__(self):
    super(DetNet, self).__init__()

    self.a1 = [nn.Linear(100, 1) for _ in range(4)]
    self.a2 = [nn.Linear(25, 1) for _ in range(16)]
    self.a3 = [nn.Linear(80, 1) for _ in range(5)]

    self.b1 = nn.Linear(4, 1)
    self.b2 = nn.Linear(16, 1)
    self.b3 = nn.Linear(5, 1)

    self.c = nn.Linear(3, 1)

    self.tanh = nn.Tanh()

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
    a1_out, a2_out, a3_out = [], [], []
    a1_in, a2_in, a3_in = self.arrange_input(x)
    for i in range(4):
      a1_out.append(self.tanh(self.a1[i](a1_in[i])))
    for i in range(16):
      a2_out.append(self.tanh(self.a2[i](a2_in[i])))
    for i in range(5):
      a3_out.append(self.tanh(self.a3[i](a3_in[i])))
    c_in = torch.cat([
      self.tanh(self.b1(torch.cat(a1_out))),
      self.tanh(self.b2(torch.cat(a2_out))),
      self.tanh(self.b3(torch.cat(a3_out)))
    ])
    return self.c(c_in)

