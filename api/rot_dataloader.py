import torch

class RotatedImageDataLoader:
  def __init__(self, dataset, device, batch_size=36):
    self.batches = []
    images = torch.empty((batch_size, 400), device=device)
    labels = torch.empty(batch_size, device=device, dtype=int)
    cnt = 0
    for image, label in dataset:
      images[cnt] = image
      labels[cnt] = label
      cnt += 1
      if cnt == batch_size:
        self.batches.append((images, labels))
        images = torch.empty((batch_size, 400), device=device)
        labels = torch.empty(batch_size, device=device, dtype=int)
        cnt = 0
    if cnt > 0:
      indices = torch.tensor([i for i in range(cnt)])
      ex_images = images.index_select(indices)
      ex_labels = labels.index_select(indices)
      self.batches.append((ex_images, ex_labels))
  
  def __iter__(self):
    return iter(self.batches)