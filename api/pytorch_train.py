from torch.utils import data
from torch import nn

from rot_dataset import RotatedImageDataset
from conv_net import ConvNet
from lin_net import LinNet

import torch.optim as optim

import signal
import torch
import sys
import os

args = sys.argv[1:]
lin = 'lin' in args
use_cuda = 'cuda' in args

model_path = './model_{}.tar'.format('lin' if lin else 'conv')
device = torch.device('cuda:0') if use_cuda else None

model = LinNet() if lin else ConvNet()
if use_cuda:
  model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if os.path.isfile(model_path):
  state = None
  if use_cuda:
    state = torch.load(model_path, map_location=device)
  else:
    state = torch.load(model_path)
  model.load_state_dict(state)

def signal_handler(sig, frame):
  print('Saving model...')
  torch.save(model.state_dict(), model_path)
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

dataset = RotatedImageDataset(250000, lin=lin, use_cuda=use_cuda)
test_dataset = RotatedImageDataset(50000, lin=lin, use_cuda=use_cuda)

dataloader = data.DataLoader(dataset, batch_size=36)
test_dataloader = data.DataLoader(test_dataset, batch_size=36)

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

torch.save(model.state_dict(), model_path)