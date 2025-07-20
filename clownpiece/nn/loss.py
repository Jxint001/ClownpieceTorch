# MSE, CrossEntropy

from clownpiece.nn.module import Module
from clownpiece import Tensor


# loss.py
class MSELoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()

  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    diff = target - input
    out_mid = diff * diff

    return out_mid.reshape(-1).mean(dim=0)

class CrossEntropyLoss(Module):
  def __init__(self, reduction: str = 'mean'):
    super().__init__()
    
  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    # logits is of shape (..., num_class)
    # target is of shape (...), and it's value indicate the index of correct label
    if logits.shape[:-1] != target.shape:
          raise ValueError("shape mismatch in CrossEntropyLoss")

    logits = logits.reshape([-1, logits.shape[-1]])
    target = target.reshape(-1)

    log_probs = logits - logits.exp().sum(dim=-1, keepdims=True).log()
    
    loss_list = []
    for i in range(logits.shape[0]):
      class_idx = int(target[i].item())
      loss_list.append(log_probs[i, class_idx])
        
    return -sum(loss_list) / logits.shape[0]
