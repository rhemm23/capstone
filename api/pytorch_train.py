from torch.utils import data
from torch import nn

from rot_dataset import RotatedImageDataset
from conv_net import ConvNet
from lin_net import LinNet

from torch_model_loader import TorchModelLoader
from json_model_loader import JsonModelLoader
from bin_model_loader import BinModelLoader

import torch.optim as optim

import signal
import torch
import json
import sys
import os

args = sys.argv[1:]
device = torch.device('cuda:0' if ('cuda' in args) else 'cpu')
model_path = './model'

loader = None
if 'json' in args:
  model_path += '.json'
  loader = JsonModelLoader()
elif 'bin8' in args:
  model_path += '8.bin'
  loader = BinModelLoader(8)
elif 'bin16' in args:
  model_path += '16.bin'
  loader = BinModelLoader(16)
else:
  model_path += '.tar'
  loader = TorchModelLoader()

model = LinNet()
model.to(device)

if os.path.isfile(model_path):
  loader.load(model_path, model, device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def save_model():
  loader.save(model.state_dict(), model_path)

def signal_handler(sig, frame):
  print('Saving model...')
  save_model()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

dataset = RotatedImageDataset(1080, device=device)
test_dataset = RotatedImageDataset(1080, device=device)

dataloader = data.DataLoader(dataset, batch_size=36)
test_dataloader = data.DataLoader(test_dataset, batch_size=36)

for i in range(10000):
  batch_cnt = 0
  tot_loss = 0
  model.train()
  for input, target in dataloader:
    batch_cnt += 1
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    tot_loss += loss.item()

  if i % 100 == 0:
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