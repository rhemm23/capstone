import torch

class BinModelLoader:
  def save(self, state_dict, path):
    torch.save(state_dict, path)

  def load(self, path, device):
    return torch.load(path, map_location=device)