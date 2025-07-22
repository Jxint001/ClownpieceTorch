from typing import Callable, List, Any, Union
import os
from PIL import Image
import numpy as np
import csv

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
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.load_data()
        # print("data is ", self.data)

    def load_data(self):
        # read CSV and store transformed rows
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if self.transform:
                    self.data.append(self.transform(row))
                else:
                    self.data.append(row)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

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
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.load_data()

    def load_data(self):
        # 1. read the subdirectories
        classes = sorted([d for d in os.listdir(self.file_path) if os.path.isdir(os.path.join(self.file_path, d))])
        # 2. assign label_id for each subdirectory (i.e., class label)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # 3. read files in subdirectory
        for cls_name in classes:
            cls_dir = os.path.join(self.file_path, cls_name)

            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    img = Image.open(fpath).convert('RGB')

                    # 4. convert PIL Image to np.ndarray
                    img_np = np.array(img)

                    # 5. apply transform
                    if self.transform:
                        img_np = self.transform(img_np)

                    # 6. store transformed image and label_id
                    self.data.append(img_np)
                    self.labels.append(self.class_to_idx[cls_name])

    def __getitem__(self, index):
        # index->(image, label_id)
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
  
"""
Image Transforms
"""

# These are functions that return desired transforms
#   args -> (np.ndarray -> np.ndarray or Tensor)
def sequential_transform(*trans):
    def func(x):
        for t in trans:
            x = t(x)
        return x
    return func

def resize_transform(size):
    def func(img):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size)
        return np.array(pil_img)
    return func

def normalize_transform(mean, std):
    def func(img):
        img = img.astype(np.float32) / 255.0
        return (img - mean) / std
    return func

def to_tensor_transform():
    def func(img):
        return Tensor(img.tolist())
    return func