from torch import nn

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(1, 64, 3)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    self.lin1 = nn.Linear(1024, 256)
    self.lin2 = nn.Linear(256, 128)
    self.lin3 = nn.Linear(128, 36)

    self.maxpool = nn.MaxPool2d(2, stride=2)
    self.flatten = nn.Flatten()

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.25)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.maxpool(self.relu(self.conv1(x)))
    x = self.maxpool(self.relu(self.conv2(x)))
    x = self.dropout(x)
    x = self.flatten(x)
    x = self.relu(self.lin1(x))
    x = self.relu(self.lin2(x))
    x = self.dropout(x)
    return self.softmax(self.lin3(x))