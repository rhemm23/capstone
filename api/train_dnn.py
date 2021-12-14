from torch import nn

from rot_dataloader import RotatedImageDataLoader, TestImageDataLoader
from rot_dataset import RotatedImageDataset, TestImageDataset

from lin_net import RotNet, DetNet

from torch_model_loader import TorchModelLoader
from json_model_loader import JsonModelLoader
from bin_model_loader import BinModelLoader

from pymongo import MongoClient

import torch.optim as optim
import numpy as np

import signal
import torch
import math
import sys
import os

def bcau(sub_image):
  for y in range(5):
    for x in range(5):
      avg = 0
      for i in range(4):
        for j in range(4):
          avg += sub_image[(y * 4) + i][(x * 4) + j]
      avg = avg // 16
      for i in range(4):
        for j in range(4):
          value = sub_image[(y * 4) + i][(x * 4) + j]
          new_value = (value + 32) if (value > avg) else (value - 32)
          new_value = max(0, min(255, new_value))
          sub_image[(y * 4) + i][(x * 4) + j] = new_value

def iru(sub_image, rnn_out):
  theta = 0
  for i in range(36):
    if rnn_out[i]:
      theta = math.radians(i * 10)
      break
  res_image = [0 for _ in range(400)]
  for y in range(20):
    for x in range(20):
      nx = int(math.cos(theta) * (x - 10) - math.sin(theta) * (y - 10)) + 10
      ny = int(math.sin(theta) * (x - 10) + math.cos(theta) * (y - 10)) + 10
      if nx >= 0 and ny >= 0 and nx < 20 and ny < 20:
        res_image[(ny * 20) + nx] = sub_image[(y * 20) + x]
  for i in range(400):
    sub_image[i] = res_image[i]

def heu(sub_image):
  cnt = [0 for _ in range(256)]
  for pixel in sub_image:
    cnt[pixel] += 1
  cdf = [cnt[0]]
  for i in range(1, 256):
    cdf.append(cdf[-1] + cnt[i])
  for i in range(400):
    sub_image[i] = int(cdf[sub_image[i]] * 256 / 400)

model_path = './current_det.json'
device = torch.device('cpu')

client = MongoClient()
db = client.capstone

model = DetNet()
loader = JsonModelLoader()

if os.path.isfile(model_path):
  loader.load(model_path, model, device)

batchs = []
cnt = 0
tcnt = 0

d = torch.empty(15, 400, dtype=torch.float)
l = torch.empty(15, 1, dtype=torch.float)

faces = list(db.prepped_data.find({ 'label': 1 }))
query = db.prepped_data.find({ 'label': 0 })
all_images = []

while True:
  next_non_face = []
  for _ in range(47):
    next_non = next(query, None)
    if not next_non:
      break
    next_non_face.append(next_non)
  if len(next_non_face) == 47:
    all_images += faces
    all_images += next_non_face
  else:
    break

for temp in all_images:
  tcnt += 1
  print(tcnt)
  data = np.frombuffer(temp['data'], dtype=np.uint8).tolist()
  for i in range(400):
    d[cnt][i] = data[i] / 256
  l[cnt][0] = float(temp['label'])
  cnt += 1
  if cnt == 15:
    cnt = 0
    batchs.append((d, l))
    d = torch.empty(15, 400, dtype=torch.float)
    l = torch.empty(15, 1, dtype=torch.float)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def save_model():
  loader.save(model_path, model)

def signal_handler(sig, frame):
  print('Saving model...')
  save_model()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

for i in range(100):
  print('Epoch {}'.format(i))
  batch_cnt = 0
  tot_loss = 0
  model.train()
  for input, target in batchs:
    batch_cnt += 1
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    tot_loss += loss.item()

  cnt = 0
  pass_cnt = 0

  model.eval()
  for input, target in batchs:
    output = model(input).tolist()
    target = target.tolist()
    for i in range(len(output)):
      cnt += 1
      temp_target = 1 if target[i][0] == 1.0 else 0
      temp_output = 1 if output[i][0] >= 0.75 else 0
      pass_cnt += 1 if temp_target == temp_output else 0

  print('Passing tests: {} / Total tests: {}'.format(pass_cnt, cnt))
  print('Accuracy: {}%'.format(round(pass_cnt / cnt * 100)))

save_model()

# rnn = RotNet()
# device = torch.device('cpu')

# loader = JsonModelLoader()
# loader.load('./rnw.json', rnn, device)

# dataset = []
# img_count = 0
# for image in db.demo_images.find():
#   img_count += 1
#   print(img_count)
#   for sub_image in image['sub_images']:
#     data = np.frombuffer(sub_image['data'], dtype=np.uint8).tolist()
#     heu(data)
#     tens = torch.empty(400, dtype=torch.double).to(device=device)
#     for i in range(400):
#       tens[i] = data[i] / 256
#     prediction = rnn.get_prediction(tens)
#     iru(data, prediction)
#     data = np.array(data, dtype=np.uint8).reshape((20, 20)).tolist()
#     bcau(data)
#     data = np.array(data, dtype=np.uint8).reshape((400,))
#     # tens = torch.empty(400, dtype=torch.double).to(device=device)
#     # for i in range(400):
#     #   tens[i] = data[i] / 256
#     # label = torch.tensor(1 if sub_image['is_face'] else 0).to(device=device)
#     db.prepped_data.insert_one({
#       'data': data.tobytes(),
#       'label': 1 if sub_image['is_face'] else 0
#     })
#     # dataset.append((tens, label))

# print(len(dataset))
# exit()
