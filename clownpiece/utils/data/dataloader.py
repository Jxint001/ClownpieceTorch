from clownpiece.utils.data.dataset import Dataset

class DefaultSampler:
  def __init__(self, length, shuffle):
    pass
  def __iter__(self):
    pass
    
  def __len__(self):
    pass
  
def default_collate_fn(batch):
  pass

class Dataloader:
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 drop_last=False, 
                 sampler=None, 
                 collate_fn=None):
        pass        
    def __iter__(self):
        # yield a batch of data
        pass

    def __len__(self):
        # number of batches, not the number of items in dataset
        pass