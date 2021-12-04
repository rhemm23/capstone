from qtorch.auto_low import sequential_lower

from lin_net import LinNet

from torch_model_loader import TorchModelLoader
from json_model_loader import JsonModelLoader
from bin_model_loader import BinModelLoader

import qtorch
import torch
import sys

forward_num = qtorch.FixedPoint(wl=16, fl=10)

device = torch.device('cpu')
model = sequential_lower(
  LinNet(),
  layer_types=['linear'],
  forward_number=forward_num
)

out_path = None
out_loader = None
if sys.argv[1] == 'json':
  out_path = './model.json'
  out_loader = JsonModelLoader()
elif sys.argv[1] == 'bin8':
  out_path = './model8.bin'
  out_loader = BinModelLoader(8)
elif sys.argv[1] == 'bin16':
  out_path = './model16.bin'
  out_loader = BinModelLoader(16)
elif sys.argv[1] == 'torch':
  out_path = './model.tar'
  out_loader = TorchModelLoader()

parts = sys.argv[2].split('.')
from_loader = None
if parts[-1] == 'json':
  from_loader = JsonModelLoader()
elif parts[-1] == 'tar':
  from_loader = TorchModelLoader()
elif parts[-1] == 'bin':
  if parts[-2].endswith('8'):
    from_loader = BinModelLoader(8)
  elif parts[-2].endswith('16'):
    from_loader = BinModelLoader(16)

from_loader.load(sys.argv[2], model, device)
out_loader.save(out_path, model)
