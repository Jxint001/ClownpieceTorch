from typing import Callable, List, Any, Union
import os
from PIL import Image
import numpy as np

from clownpiece.tensor import Tensor

class Dataset:
  
  def __init__(self):
    pass

  def __getitem__(self, index):
    """
    Returns the item at the given index.
    """
    raise NotImplementedError("Dataset __getitem__ method not implemented")
  
  def __len__(self):
    """
    Returns the total number of item
    """
    raise NotImplementedError("Dataset __len__ method not implemented")
  
"""
CSV
"""
  
class CSVDataset(Dataset):

    file_path: str
    data: List[Any]
    transform: Callable

    def __init__(self, file_path: str, transform: Callable = None):
        # load CSV, apply transform
        pass

    def load_data(self):
        # read CSV and store transformed rows
        # should be called at the end of __init__
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

"""
Image
"""

class ImageDataset(Dataset):

    file_path: str
    data: List[Union[np.ndarray, Tensor]]
    labels: List[int]
    transform: Callable
    class_to_idx: dict[str, int]

    def __init__(self, file_path: str, transform: Callable = None):
        pass

    def load_data(self):
        # 1. read the subdirectories
        # 2. assign label_id for each subdirectory (i.e., class label)
        # 3. read files in subdirectory
        # 4.    convert PIL Image to np.ndarray
        # 5.    apply transform
        # 6.    store transformed image and label_id
        pass

    def __getitem__(self, index):
        # index->(image, label_id)
        pass

    def __len__(self):
        pass
  
"""
Image Transforms
"""

# These are functions that return desired transforms
#   args -> (np.ndarray -> np.ndarray or Tensor)
def sequential_transform(*trans):
    pass

def resize_transform(size):
    pass

def normalize_transform(mean, std):
    pass

def to_tensor_transform():
    pass