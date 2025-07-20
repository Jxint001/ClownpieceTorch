# Linear, Embedding, LayerNorm, BatchNorm, MultiheadAttention

from typing import Optional
from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module, Parameter, Buffer
from . import init
import math



class Linear(Module):

  in_features: int
  out_features: int
  weight: Tensor

  def __init__(self, in_features: int, out_features: int, bias: bool=True):
    # remember to wrap W and b in Parameter class, otherwise they won't be registered.
    # for now, init W, b with empty
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(Tensor.empty([out_features, in_features]))
    if bias:
      self.bias = Parameter(Tensor.empty([out_features]))
    else:
      self.register_parameter("bias", None)

    self.reset_parameters()

  def reset_parameters(self) -> None:
    bound = math.sqrt(1 / self.in_features)
    init.uniform_(self.weight, -bound, bound)
    
    if self.bias is not None:
      init.uniform_(self.bias, -bound, bound)

  def forward(self, x: Tensor) -> Tensor:
    # print("before linear: ", x)
    # print(str(self))
    ret = x.matmul(self.weight.transpose(-1, -2)) + self.bias
    # print("after linear: ", ret)
    return ret

  def extra_repr(self):
    return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
    def __init__(self, num_embd: int, embd_dim: int):
      super().__init__()
      self.num_embd = num_embd
      self.embd_dim = embd_dim
      self.weight = Parameter(Tensor.empty([num_embd, embd_dim]))
      self.reset_parameters()

    def reset_parameters(self) -> None:
      init.normal_(self.weight, 0, 1)

    def forward(self, x: Tensor) -> Tensor:
      return self.weight[x]

    def extra_repr(self):
      return f"num_embd={self.num_embd}, embd_dim={self.embd_dim}"
    
class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
      # input is reshaped to (-1, num_features) for normalziation.
      # for example:
      #   to normalize last two dimensions of tensor (batch_size, height, width)
      #   then num_features should be height x width
      # this interface differs from pytorch
      super().__init__()
      self.num_features = num_features
      self.eps = eps
      self.affine = affine

      if affine:
        self.gama = Parameter(Tensor.ones([num_features, ]))
        self.beta = Parameter(Tensor.zeros([num_features, ]))
      else:
        self.gama = self.register_parameter("gama", None)
        self.beta = self.register_parameter("beta", None)

    def forward(self, x: Tensor) -> Tensor:
      # print("before op x is ", x)
      fornorm = x.reshape([-1, self.num_features])
      # print("fornorm is ", fornorm)
      mean = fornorm.mean(dim=-1, keepdims=True)
      # print("mean is ", mean)
      var = fornorm.var(dim=-1, keepdims=True)
      # print("var is ", var)
      x_hat = (x - mean) / (var + self.eps).sqrt()
      # print("x_hat is ", x_hat)
      if self.affine:
        x_hat = x_hat * self.gama + self.beta
      
      ret = x_hat.reshape(list(x.shape))
      # print("ret is ", ret)
      return ret

    def extra_repr(self):
      return f"num_features={self.num_features}, eps={self.eps}, affine={self.affine}"

class BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
      super().__init__()
      self.num_features = num_features
      self.eps = eps
      self.momentum = momentum
      self.affine = affine

      if affine:
        self.gama = Parameter(Tensor.ones([num_features, ]))
        self.beta = Parameter(Tensor.zeros([num_features, ]))
      else:
        self.gama = self.register_parameter("gama", None)
        self.beta = self.register_parameter("beta", None)

      self.running_mean = Buffer(Tensor.zeros([num_features, ]))
      self.running_var = Buffer(Tensor.ones([num_features, ]))

    def forward(self, x: Tensor) -> Tensor:
      if self.training:
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
      else:
        mean = self.running_mean
        var  =self.running_var
      
      x_hat = (x - mean) / (var + self.eps).sqrt()
      if self.affine:
        x_hat = x_hat * self.gama + self.beta
      return x_hat

    def extra_repr(self):
      return f"num_features={self.num_features}, eps={self.eps}, affine={self.affine}, momentum={self.momentum}"
    
class MultiheadAttention(Module):
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.num_heads = num_heads
      self.head_dim = hidden_dim // num_heads

      if self.head_dim * num_heads != hidden_dim:
          raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
      
      self.q_proj = Linear(hidden_dim, hidden_dim, bias=bias)  # Query projection
      self.k_proj = Linear(hidden_dim, hidden_dim, bias=bias)  # Key projection
      self.v_proj = Linear(hidden_dim, hidden_dim, bias=bias)  # Value projection
      self.out_proj = Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):
      batch_size, seq_len, _ = hidden_states.shape
      
      # calculate Q、K、V
      # [batch_size, seq_len, hidden_dim]
      Q = self.q_proj(hidden_states)
      K = self.k_proj(hidden_states)
      V = self.v_proj(hidden_states)
      
      # new shape: [batch_size, num_heads, seq_len, head_dim]
      Q = Q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
      K = K.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
      V = V.reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)
      
      # output shape: [batch_size, num_heads, seq_len, seq_len]
      attn_scores = Q.matmul(K.transpose(-1, -2)) / math.sqrt(self.head_dim)

      if attn_mask is not None:
        attn_scores = attn_scores + attn_mask # * (float('-inf'))
      
      attn_weights = attn_scores.softmax(dim=-1)
      attn_output = attn_weights.matmul(V)

      # to old shape: [batch_size, seq_len, num_heads, head_dim]
      attn_output = attn_output.transpose(1, 2)
      
      # [batch_size, seq_len, hidden_dim]
      attn_output = attn_output.reshape([batch_size, seq_len, self.hidden_dim])
      
      output = self.out_proj(attn_output)
      return output

    def extra_repr(self):
      return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"