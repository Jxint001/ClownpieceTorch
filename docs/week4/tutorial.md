# Clownpiece-Torch Week 4: Bringing It All Together

Congratulations on reaching the final week of the Clownpiece-Torch series! In Week 4, we will complete our deep learning library with periphral utilies:

- **Data loading utilities**: `Dataset`, `CSVDataset`, `ImageDataset`, `Dataloader`, and image transforms
- **Optimization algorithms**: `Optimizer`, `SGD`, `Adam`
- **Learning rate schedulers**: `LRScheduler`, `LambdaLR`, `ExponentialLR`, `StepLR`

By the end of this tutorial, you will understand how to structure and manage data, implement training loops with custom optimizers, and control learning rates during training. These utilies make the framework more convenient and versatile.

As a good tradition, last week's workload is about half of the previous.

---

# Section 1: Dataset and Dataloader

## 1.1 Dataset Abstraction

We begin by defining a common interface for datasets. Our `Dataset` base class resides in `clownpiece/utils/data/dataset.py` 

> to avoid naming conflicts, we have renamed `utils.py` to `utils_.py`.

It provides two key interfaces:
- `dataset[index]` to retrieve item at index. 
    - The index may or may not be integer based on specific use case. 
    - The value may, or may not be (a tuple of) Tensors.
    - You may consider dataset as a dictionary tailored for dataloading.
- `len(dataset)` to get the total number of items.

```python
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
```

> There is another less-common type of dataset called 'iterable-dataset', for scenarios where random access with index is slow or even impossible. (e.g., streaming of data from remote server)

## 1.2 CSVDataset

`CSVDataset` reads rows from a CSV file and applies an optional transform to each row:

```python
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
```

**Usage Example:**

```python
from clownpiece.utils.data import CSVDataset

dataset = CSVDataset('data/values.csv', 
                    transform=lambda row: Tensor([float(x) for x in row]))
print(len(dataset), dataset[0])
```

This initilializes dataset to read values.csv, and concatenates each row into a 1D Tensor. User may alter the transform, for example, to split last column of each row as label, and the others as features.

## 1.3 ImageDataset

`ImageDataset` loads images organized in subdirectories where each subdirectory corresponds to a class label.



```python
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
```

Arguments:

- `file_path` points to the root directory of dataset
- `transform` is a callable that apply transforms

Image dataset assigns an integer label id to string class label in range $[0, \text{num\_class})$.

- `self.class_to_idx`: Dict[str, int] maps the string to integer id. 

We also provides some common transforms on images:
```python
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
```

**Usage Example:**

```bash
# File Structure
animal_dataset/
|- cat/
|   |- 0.png
|   |- 1.png
|   |- ...
|- dog/
|   |- 100.png
|   |- ...
|- fish/
|   |- 200.png
|   |- ...
|...
```

```python
from clownpiece.utils.data import ImageDataset, sequential_transform, resize_transform, normalize_transform, to_tensor_transform

transform = sequential_transform(
    resize_transform((28, 28)),
    normalize_transform(0.5, 0.2),
    to_tensor_transform()
)
dataset = ImageDataset('data/animal_dataset', transform=transform)
```

This code snippet:
1. reads animal_dataset directory
2. sets class_to_index to $\begin{cases}\text{cat} \to 0 \\ \text{dog} \to 1 \\ \text{fish} \to 2 \end{cases}$ (this mapping can be reordered, and is fine)
3. transforms the image by resizing, normalizing, and converting to Tensor
4. stores images and label ids in pair

## 1.4 Dataloader

`Dataloader` wraps a `Dataset` to provide batch iteration, optional shuffling, and collating:

```python
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
```

Arguments:

- `dataset`: an instance of Dataset
- `batch_size`: the number of items in each batch
- `shuffle`: whether to shuffle indices at each epoch
- `drop_last`: if the total number of items in dataset is not divisible by batch size, whether to the last batch (the remainder).
- `sampler`: custom sampler to perform shuffle. `sampler` argument is mutually exclusive with `shuffle`.
- `collate_fn`: specify how to collate a list of samples from dataset to form a batch. (e.g., stack a list of tensor to a single batch tensor)

Other details:
### Sampler

A sampler is an iterable class that provides `__iter__` and `__len__`.

- `__iter__` yields index, one at a time, into the dataset,
- `__len__` returns the total number of indices the `__iter__` will yield.

When `__iter__` is hausted once, the dataset is iterated for one epoch. The `__iter__` can yield only a subset of indices, which is common for distributed dataloader where each worker processes a different subset of dataset.

When `sampler is None`, you should provide a default sampler that yeilds all indices (or a shuffle of all indices) in the dataset depending on the `shuffle` argument.

### collate_fn

The `collate_fn` defines how to collate a list of samples in a batch (obtained by indexing the dataset with sampler) into a batched data at `__iter__` method of `Dataloader`.

For example, by accessing `ImageDataset` with `batch size = 3`, you will get something like:
```python
list_of_samples = [
    (Tensor 28x28, int), # 28x28 image + label_id
    (Tensor 28x28, int),
    (Tensor 28x28, int)
]
```

The desired batched data is `(Tensor 3x28x28, Tensor 3)`. 

When `collate_fn is None`, you should provide a default collate_fn, that:

- if each sample is not a tuple:
    - if sample is Tensor, stack the Tensors at dim 0
    - if sample is scalar, convert to Tensor then stack
- if each sample is a tuple:
    - apply the same stacking rule for each element in tuple as above.

**Usage Example:**

```python
from clownpiece.utils.data import Dataloader 

loader = Dataloader(dataset, batch_size=32, shuffle=True)
for X_batch, y_batch in loader:
    # training code here
```

---

# Section 2: Optimizer

In Week 3, we saw how gradients flow through our models. Now we implement *Optimizers* to update learnable parameters based on those gradients. All optimizers inherit from the `Optimizer` base class in `clownpiece/utils/optim/optimizer.py`.

## 2.1 Optimizer Base Class

```python
class Optimizer:
    param_groups: List[Dict[str, Any]]       # list of parameter groups
    state: Dict[Parameter, Dict[str, Any]]   # mapping param_id -> optimizer state
    defaults: Dict[str, Any]                 # default hyperparams for each group

    def __init__(self, parameters: Iterable[Parameter] | Iterable[Dict], defaults: Dict[str, Any]):
        """
        - `parameters`: an iterable of `Parameter` objects or a list of dicts defining parameter groups.
            - if iterable of `Parameter`, add it as the first param_group.
        - `defaults`: a dict of default hyperparameters (e.g., learning rate).
        """
        pass

    def add_param_group(self, param_group: Dict[str, Any]):
        """Merge defaults into `param_group` and add to `param_groups`."""
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        self.param_groups.append(param_group)

    def step(self):
        """Perform a single optimization step (update all parameters).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def zero_grad(self, set_to_None: bool = True):
        """Reset gradients for all parameters."""
        pass
```

**Key concepts:**
- `param_groups`: lists of parameters and their specific hyperparameters (allow different groups to have different learning rates).
    - by convention, in param group, `'params'` field is a list of parameters, `'lr'` is the learning rate.
- `state`: internal buffers keyed by `Parameter.param_id` (e.g., momentum terms, moment estimates).
- `defaults`: the default hyperparameters applied when adding new groups.

You need to assign `param_id` to `Parameter` (unique across param groups) at `__init__`. (Additionally, you can define `__hash__` and `__eq__` for `Parameter` class based on `param_id` to use `Parameter` key directly.)

## 2.2 SGD (Stochastic Gradient Descent)

**SGD** is a first-order optimization algorithm used to minimize the loss function in machine learning models. It updates parameters using an estimate of the gradient based on a small, randomly sampled minibatch of data rather than the full dataset — **this is where the "stochasticity" lies**. In practice, this sampling is done by partitioning dataset into batches.

```python
class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
        pass

    def step(self):
        pass
```

### **Mathematical Formulation**

Let:

* $\eta$: learning rate
* $\mu$: momentum coefficient
* $\gamma$: damping coefficient
* $\lambda$: weight decay
* $v_t$: velocity
* $\nabla L(\theta_t)$: gradient at $t$

---
#### 1. **Vanilla SGD** (no momentum, no weight decay):

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta \nabla L(\theta_t)
\end{aligned}
$$

---

#### 2. **SGD with Weight Decay** (L2 regularization):

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta \left( \nabla L(\theta_t) + \lambda \theta_t \right)
\end{aligned}
$$

---

#### 3. **SGD with Momentum (and Damping):**

$$
\begin{aligned}
v_t &= \mu v_{t-1} + (1 - \gamma) \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_t
\end{aligned}
$$

---

#### 4. **SGD with Momentum + Weight Decay + Damping**:

$$
\begin{aligned}
v_t &= \mu v_{t-1} + (1 - \gamma) \left( \nabla L(\theta_t) + \lambda \theta_t \right) \\
\theta_{t+1} &= \theta_t - \eta v_t
\end{aligned}
$$


## 2.3 Adam (Adaptive Moment Estimation)

**Adam** is an adaptive optimization algorithm that combines the benefits of **momentum** and **RMSProp**. It computes **exponentially decaying averages** of past gradients (first moment) and squared gradients (second moment), and uses these to scale the learning rate adaptively for each parameter.

Unlike SGD, Adam adapts the learning rate **per parameter**, based on its update history. It also uses **bias correction** to counteract the initial moment estimates being close to zero.

---
```python
class Adam(Optimizer):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    pass

  def step(self):
    pass

```
---

### **Mathematical Formulation**

Let:

* $\theta_t$: the parameters (weights) of the model at time step $t$
* $L(\theta_t)$: the loss function
* $g_t = \nabla_\theta L(\theta_t)$: the **gradient of the loss** w\.r.t. parameters at step $t$. If with weight decay $\lambda$, then $g_t = \nabla_\theta L(\theta_t) + \lambda \theta_t$.

This is the raw gradient used to update parameters — it points in the direction of steepest ascent (Adam minimizes it).

---


#### 1. **First Moment (Mean) — $m_t$**

This is the **exponentially decaying average of past gradients**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

* It smooths the gradients over time, similar to **momentum**.
* $\beta_1 \in [0, 1)$: typically 0.9.
* Interpreted as an estimate of the **expected gradient** $\mathbb{E}[g_t]$.

#### 2. **Second Moment (Uncentered Variance) — $v_t$**

This tracks the **exponentially decaying average of the squared gradients**:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

* It estimates the **uncentered variance** $\mathbb{E}[g_t^2]$.
* $\beta_2 \in [0, 1)$: typically 0.999.
* Controls how aggressively the learning rate is scaled for each parameter.

---

### **Bias Correction**

Since both $m_0$ and $v_0$ are initialized as zero vectors, they are biased toward zero in early steps. To correct for this:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

These are the **bias-corrected estimates** of the first and second moments.

---

### **Final Update Rule**

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

* $\eta$: learning rate
* $\epsilon$: small constant to avoid division by zero (e.g., $10^{-8}$)

Each parameter is updated with an **adaptive learning rate**, scaled inversely by the square root of its estimated variance — making Adam both **scale-invariant** and **stable**.

---
# Section 3: Learning Rate Scheduler

Scheduling the learning rate can improve training stability and convergence. The `LRScheduler` class in `clownpiece/utils/optim/lr_scheduler.py` provides a framework:

## 3.1 LRScheduler Base Class

```python
class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        # You need to store base_lrs as to allow closed-form lr computation; you may assume that, optimizer does not add new param groups after initializing the LRSchduler.
        pass

    def get_lr(self) -> List[float]:        
        # Calculate new learning rates for each param_group.
        # To be implemented by subclasses with a closed-form formula based on self.last_epoch.
        raise NotImplementedError

    def step(self, epoch: int = None):
        # Advance to `epoch` (or next epoch if None), then update optimizer.param_groups:
        # 1. Update self.last_epoch 
        # 2. Compute new LRs via self.get_lr()
        # 3. Assign each param_group['lr'] = corresponding new LR
        pass

    def get_last_lr(self) -> List[float]:
        # Return the most recent learning rate for each param_group.
        pass
```

### Key Points
- **step** is fully implemented in the base class, applying whatever `get_lr` returns.
- **get_lr** is abstract: subclasses override it to provide a closed-form schedule based on `self.last_epoch`. (the name of last epoch is confusing, it's better referred to as current epoch.)
- **get_last_lr** simply reads back the current learning rates from the optimizer.

## 3.2 LambdaLR

`LambdaLR` scales each base learning rate by a user-provided function `lr_lambda(epoch)`:

```python
class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda: Callable[[int], float], last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass
```

Use case: exponential decay via `lr_lambda = lambda e: 0.95**e`, or any custom shape.

## 3.3 ExponentialLR

`ExponentialLR` decays the learning rate by a fixed factor `gamma` each epoch:

```python
class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass

```

Example: `gamma=0.9` reduces LR by 10% each epoch.

## 3.4 StepLR


The `StepLR` is similar to `ExponentialLR` except it applies the lr update every `step_size` epochs.

```python
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass
```

---

With these schedulers, you can plug-in any optimizer and automatically adjust its learning rate according to your chosen schedule. Happy scheduling!

> Remark: in recent pytorch version, the closed-form lr computation is deprecated, and the code is transfering to use progressive step (epoch $\to$ epoch + 1) only.

---

# Grading

First, as usual, make sure to pass `tests/week4/grade_part{i}.py` and `grade_all.py`. 

After that, proceed to `tests/week4/estate_value_predict` and `fashion-minist/`, where we rewrite the similar training tasks from week3 with our own utility.

---

# Final Report:

This time, besides a report for week4, you should also prepare a presentation PPT to summarize your work and gains during the project.