from pymongo import MongoClient
from torch.utils import data
from torch import nn

import torch.optim as optim
import numpy as np

import signal
import torch
import sys
import os

MODEL_PATH = './model.tar'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

client = MongoClient()
db = client.capstone

class RotatedImageDataset(data.IterableDataset):
  def __init__(self, count=None):
    self.count = count

  def __next__(self):
    rot = next(self.query, None)
    if rot:
      data = np.frombuffer(rot['data'], dtype=np.uint8).tolist()
      for i in range(400):
        data[i] = data[i] / 256
      dat = torch.tensor([np.array(data).reshape((20, 20)).tolist()]).to(device=device)
      lab = torch.tensor(int(rot['rotation'] / 10)).to(device=device)
      return (dat, lab)
    else:
      raise StopIteration

  def __iter__(self):
    self.query = db.rot_data.find()
    self.query = self.query.limit(self.count) if self.count else self.query
    return self

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3, padding=1)

    self.lin1 = nn.Linear(256, 64)
    self.lin2 = nn.Linear(64, 36)

    self.maxpool = nn.MaxPool2d(2, stride=2)
    self.flatten = nn.Flatten()

    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.maxpool(self.relu(self.conv1(x)))
    x = self.maxpool(self.relu(self.conv2(x)))
    x = self.flatten(x)
    x = self.relu(self.lin1(x))
    return self.softmax(self.lin2(x))

model = Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if os.path.isfile(MODEL_PATH):
  checkpoint = torch.load(MODEL_PATH, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def save_model():
  state = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
  }
  torch.save(state, MODEL_PATH)

def signal_handler(sig, frame):
  print('Saving model...')
  save_model()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

dataset = RotatedImageDataset()
test_dataset = RotatedImageDataset(5000)

dataloader = data.DataLoader(dataset, batch_size=50)
test_dataloader = data.DataLoader(test_dataset, batch_size=50)

for i in range(10):
  batch_cnt = 0
  tot_loss = 0
  model.train()
  for input, target in dataloader:
    batch_cnt += 1
    if batch_cnt % 1000 == 0:
      print('Epoch {} Batch: {}'.format(i, batch_cnt))
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    tot_loss += loss.item()

  print('Epoch {} Loss: {}'.format(i, tot_loss))

  cnt = 1
  pass_cnt = 0

  model.eval()
  for input, target in test_dataloader:
    output = model(input).tolist()
    target = target.tolist()
    for i in range(len(output)):
      cnt += 1
      pass_cnt += 1 if target[i] == output[i].index(max(output[i])) else 0

  print('Accuracy: {}%'.format(round(pass_cnt / cnt * 100)))

save_model()
