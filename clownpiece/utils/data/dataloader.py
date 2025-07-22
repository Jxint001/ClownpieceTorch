from clownpiece.utils.data.dataset import Dataset
import random

class DefaultSampler:
  def __init__(self, length, shuffle):
    self.length = length
    self.shuffle = shuffle

  def __iter__(self):
    indices = list(range(self.length))
    if self.shuffle:
        random.shuffle(indices)
    return iter(indices)
    
  def __len__(self):
    return self.length
  
def default_collate_fn(batch):
  if isinstance(batch[0], (tuple, list)):
    transposed = list(zip(*batch))
    return tuple(default_collate_fn(list(b)) for b in transposed)
  else:
    from clownpiece.tensor import Tensor
    # Tensor or scalar
    if hasattr(batch[0], 'stack'):
      return Tensor.stack(batch, dim=0)
    else:
      return Tensor(batch)

class Dataloader:
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 drop_last=False, 
                 sampler=None, 
                 collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler if sampler is not None else DefaultSampler(len(dataset), shuffle)
        self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn
      
    def __iter__(self):
        # yield a batch of data
        batch = []
        for idx in self.sampler:
          batch.append(self.dataset[idx])
          if len(batch) == self.batch_size:
            yield self.collate_fn(batch)
            batch = []
        if batch and not self.drop_last:
          yield self.collate_fn(batch)

    def __len__(self):
        # number of batches, not the number of items in dataset
        total = len(self.sampler)
        if self.drop_last:
          return total // self.batch_size
        else:
          return (total + self.batch_size - 1) // self.batch_size