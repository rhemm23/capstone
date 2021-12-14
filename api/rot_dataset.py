from pymongo import MongoClient
from torch.utils import data

import numpy as np
import torch

client = MongoClient()
db = client.capstone

def heu(sub_image):
  cnt = [0 for _ in range(256)]
  for pixel in sub_image:
    cnt[pixel] += 1
  cdf = [cnt[0]]
  for i in range(1, 256):
    cdf.append(cdf[-1] + cnt[i])
  for i in range(400):
    sub_image[i] = int(cdf[sub_image[i]] * 256 / 400)

class TestImageDataset(data.IterableDataset):
  def __init__(self, device=None):
    self.device = device

  def __iter__(self):
    query = db.demo_images.find({}, projection={ 'sub_images': 1 })
    results = []
    for image in query:
      for sub_image in image['sub_images']:
        if sub_image['is_face']:
          data = np.frombuffer(sub_image['data'], dtype=np.uint8).tolist()
          heu(data)
          for i in range(400):
            data[i] = data[i] / 256
          dat = torch.tensor(data, dtype=torch.double).to(device=self.device)
          lab = torch.tensor(int(sub_image['rotation'])).to(device=self.device)
          results.append((dat, lab))
    return iter(results)

class RotatedImageDataset(data.IterableDataset):
  def __init__(self, count=None, device=None):
    self.device = device
    self.count = count

  def __next__(self):
    rot = next(self.query, None)
    if rot:
      data = np.frombuffer(rot['data'], dtype=np.uint8).tolist()
      for i in range(400):
        data[i] = data[i] / 256
      dat = torch.tensor(data, dtype=torch.double).to(device=self.device)
      lab = torch.tensor(int(rot['rotation'] / 10)).to(device=self.device)
      return (dat, lab)
    else:
      raise StopIteration

  def __iter__(self):
    self.query = db.rot_data.find().limit(self.count) if self.count else db.rot_data.find()
    return self
