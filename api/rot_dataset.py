from pymongo import MongoClient
from torch.utils import data
import numpy as np
import torch

client = MongoClient()
db = client.capstone

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
      dat = torch.tensor(data).to(device=self.device)
      lab = torch.tensor(int(rot['rotation'] / 10)).to(device=self.device)
      return (dat, lab)
    else:
      raise StopIteration

  def __iter__(self):
    self.query = db.rot_data.find()
    self.query = self.query.limit(self.count) if self.count else self.query
    return self