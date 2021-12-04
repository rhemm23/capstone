import torch
import json

class JsonModelLoader:
  def save(self, state_dict, path):
    with open(path, 'w+') as file:
      res = {}
      for name in state_dict:
        res[name] = state_dict[name].tolist()
      json.dump(res, file)

  def load(self, path, model, device):
    state = {}
    with open(path, 'r') as file:
      res = json.load(file)
      for name in res:
        state[name] = torch.tensor(res[name]).to(device=device)
    model.load_state_dict(state)