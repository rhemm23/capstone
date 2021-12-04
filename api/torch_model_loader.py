import torch

class TorchModelLoader:
  def save(self, path, model):
    torch.save(model.state_dict(), path)

  def load(self, path, model, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)