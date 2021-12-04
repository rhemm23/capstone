import torch

class TorchModelLoader:
  def save(self, state_dict, path):
    torch.save(state_dict, path)

  def load(self, path, model, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)