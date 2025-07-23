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
      fornorm = x.reshape([-1, self.num_features])
      mean = fornorm.mean(dim=-1, keepdims=True)
      var = fornorm.var(dim=-1, keepdims=True)
      x_hat = (x - mean) / (var + self.eps).sqrt()
      if self.affine:
        x_hat = x_hat * self.gama + self.beta
      
      ret = x_hat.reshape(list(x.shape))
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
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width):
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_height = kernel_height
      self.kernel_width = kernel_width
      
      self.weight = Parameter(Tensor.empty([out_channels, in_channels, kernel_height, kernel_width]))
      self.bias = Parameter(Tensor.empty([out_channels]))
      self.reset_parameters()

    def reset_parameters(self):
      # kaiming initialization
      # fan_in = self.in_channels * self.kernel_height * self.kernel_width
      # bound = math.sqrt(1 / fan_in)
      # init.uniform_(self.weight, -bound, bound)
      # init.uniform_(self.bias, -bound, bound)
      init.ones_(self.weight)
      init.zeros_(self.bias)

    def forward(self, x: Tensor):
      # x: input Tensor with shape[batch_size, in_channels, in_height, in_width]
      batch_size, in_channels, in_height, in_width = x.shape

      # calculate paddings
      total_padding_h = self.kernel_height - 1 
      padding_top = total_padding_h // 2
      padding_bottom = total_padding_h - padding_top  # for potential asymmetrical padding

      total_padding_w = self.kernel_width - 1
      padding_left = total_padding_w // 2
      padding_right = total_padding_w - padding_left

      # flattened_kernel.shape = [in_channels * kernel_height * kernel_width, out_channels] = [patch_dim, out_channels]
      flattened_kernel = self.weight.reshape([self.out_channels, -1]).transpose(0, 1)

      # Unfold
      # unfolded_x.shape = [batch_size, num_patches, patch_dim]
      # num_patches = out_height * out_width = in_height * in_width
      # patch_dim = in_channels * kernel_height * kernel_width
      unfolded_x = x.unfold(self.kernel_height, self.kernel_width,
                            1, 1, # stride_height and stride_width fixed to 1
                            padding_top, padding_bottom,
                            padding_left, padding_right,
                            1, 1 # dilation_height and dilation_width fixed to 1
                            )
      # MatMul
      # output_matrix.shape = [batch_size, num_patches, out_channels]
      output_matrix = unfolded_x.matmul(flattened_kernel) 

      # Reshape
      output = output_matrix.permute([0, 2, 1])
      output = output.reshape([batch_size, self.out_channels, in_height, in_width])
      # output.shape = [batch_size, out_channels, in_height, in_width]

      # add bias
      output = output + self.bias 
      return output
    
    def extra_repr(self):
      return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_height={self.kernel_height}, kernel_width={self.kernel_width}"