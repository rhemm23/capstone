from torch import nn

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
