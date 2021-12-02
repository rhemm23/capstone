from pymongo import MongoClient
from torch.utils import data
import numpy as np
import torch

client = MongoClient()
db = client.capstone

class RotatedImageDataset(data.IterableDataset):
  def __init__(self, count=None, lin=False):
    self.count = count
    self.lin = lin

  def __next__(self):
    rot = next(self.query, None)
    if rot:
      data = np.frombuffer(rot['data'], dtype=np.uint8).tolist()
      for i in range(400):
        data[i] = data[i] / 256
      dat = None
      if self.lin:
        dat = torch.tensor(data).cuda()
      else:
        dat = torch.tensor([np.array(data).reshape((20, 20)).tolist()]).cuda()
      lab = torch.tensor(int(rot['rotation'] / 10)).cuda()
      return (dat, lab)
    else:
      raise StopIteration

  def __iter__(self):
    self.query = db.rot_data.find()
    self.query = self.query.limit(self.count) if self.count else self.query
    return self