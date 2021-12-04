from torch import nn

class LinNet(nn.Module):
  def __init__(self):
    super(LinNet, self).__init__()

    self.lin1 = nn.Linear(400, 15)
    self.lin2 = nn.Linear(15, 15)
    self.lin3 = nn.Linear(15, 36)

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.tanh(self.lin1(x))
    x = self.tanh(self.lin2(x))
    return self.softmax(self.lin3(x))