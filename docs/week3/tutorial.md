# Clownpiece-torch Week 3

In Week 2, we built a powerful autograd engine capable of tracking computations and automatically calculating gradients. While this is the core of modern deep learning frameworks, writing complex models using only raw tensor operations can be cumbersome and disorganized. This week, we will build a **Module** system, inspired by PyTorch's `torch.nn.Module`, to bring structure, reusability, and convenience to our model-building process.

The module system provides a way to encapsulate parts of a neural network into reusable components. It handles the management of learnable parameters, sub-modules, and stateful buffers, allowing you to define complex architectures in a clean, object-oriented way.

A simple example in PyTorch illustrates the concept:
```python
import torch
import torch.nn as nn

# Define a custom network by subclassing nn.Module
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers as attributes. They are automatically registered.
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    # Define the forward pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)

# The module system makes it easy to inspect all parameters
print("\nNamed Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```
Outputs:
```python
SimpleNet(
  (layer1): Linear(in_features=784, out_features=128, bias=True)
  (activation): ReLU()
  (layer2): Linear(in_features=128, out_features=10, bias=True)
)

Named Parameters:
layer1.weight: torch.Size([128, 784])
layer1.bias: torch.Size([128])
layer2.weight: torch.Size([10, 128])
layer2.bias: torch.Size([10])
```

As you can see, the `nn.Module` base class provides a clean structure and automatically tracks all the learnable parameters within the nested layers.

## Unifying Computation and State

The first design philosophy is the **unified managment of tightly coupled parts**. 

A neural network layer isn't just a single function; it's a stateful computation. It has a defined transformation (the computation) and it has internal variables that persist across calls (the state). Therefore, module comes to help by organizing of *computation* and *states* together.

In implement, module elegantly organizes this into three fundamental components:

-   **Forward Pass**: It defines the transformation the module applies to its inputs. You can think of it as the mathematical function the layer represents, like a linear transformation or a convolution. The forward takes both user specified inputs and module's internal states to produce the outputs.

-   **Parameters**: These represent the learnable state of the module, often referred to as **weights**. When you 'train' a model, you are optimizing these parameters to achieve some objective (i.e., minizing a loss function, or maximizing downstream task's accuracy). Parameters account for the majority of state in a typical deep learning model.

-   **Buffers**: These represent the non-learnable state. Sometimes a module needs to keep track of data that isn't a learnable parameter, such as the running mean and variance in a batch normalization layer. They are saved along with the parameters, but they are not updated by the optimizer during backpropagation. You will only see buffers in few special modules.

Clearly, forward pass defines the computation, while parameters and buffers form the state.

> Both parameters and buffers can change across calls, so the term *non-learnable* does **NOT** imply *constant* or *immutable*. It's more of a model structrual concept: whether they can be optimized, or only for temporary storage purpose.

---

#### Example

Let's consider a `Linear` module, which performs $y=x@W^T+b$ (we will explain the reason for transpose later).

Its forward pass might be like:
```python
class Linear(Module):
  W: Tensor # shape [ouput_channel * input_channel]
  b: Tensor # shape [ouput_channel]

  def forward(self, x: Tensor) -> Tensor: # shape [... * input_channel] ->  [... * output_channel]
    W, b = self.W, self.b
    y = x @ W.transpose() + b 
    return y
```
where `@`, `transpose`, and `+` are traced by autograd engine, and dispatched into tensor library at runtime.

`self.W, self.b` are parameters, and there are no buffers in `Linear`.

---

## Modular and Hierarchical Organization

### Modularity $\to$ Simpilicity, Reusability and Flexibility

The second core design philosophy emphasizes **modularity**. Just as individual layers encapsulate their own logic and variables, these self-contained modules can be nested and connected to form intricate network architectures.

By breaking down a complex neural network into smaller, manageable modules, the design process becomes much simpler. Instead of dealing with a monolithic block of code, you can focus on developing and testing individual components. 

Moreover, modules are highly reusable. 
- Mainstream DL frameworks offer highly-optimized, well-tested implementations for common modules like Linear, Conv2d, BatchNorm. 
- Even domain-specific modules can be reused. FlashAttention, RotaryEmbedding are widely adopted in different transformer models.

Modularity also introduces immense flexibility. Suppose you're an enthusiastic researcher with an idea to alter the structure of an existing state-of-the-art model. With a modular design, you can easily swap out or introduce new components, without having to rebuild the entire network from scratch or understanding other parts' implementation detail. This iterative approach is crucial for innovation and experimentation in deep learning, especially when models are getting more and more complicated nowadays.

---
### Hierarchy $\to$ Ease of Design and System Managment

Beyond modularity, the module system is inherently **hierarchical**, which is excellent news for system designers. Higher-level modules are composed of smaller, more basic modules, but never the other way around. 

However, from a functional standpoint, there's no noticeable difference between a basic layer and a complex block; they both remain unified under the *module abstraction* with great modularity.

> Modularity and hierarchy usually contradict each other, so this example is interesting.

With hierarchical structure, we can conceptualize a module's composition as a tree, where clear parent-child relationships are defined. 

This allows a parent module to manage the states of all its children. This centralized state management is beneficial for saving, updating, or restoring the entire module's state.

---

#### Example: Unfolding a GPT-like Model

Consider a **GPT-model**. Its module structure will be like:

```python
GPTModel
├── Embedding # Projects input IDs into hidden space
├── Positional Encoding # Adds positional information
└── Transformer Blocks # Manipulates the hidden states
    ├── Transformer Block 1
    │   ├── Multi-Head Attention # Attention mechanism
    │   │   ├── Linear # for Q, K, V projections
    │   │   └── Linear # for output projection
    │   ├── Layer Normalization # Layer Norm
    │   └── Feed-Forward Network # FFN
    │       ├── Linear
    │       └── Activation
    └── Transformer Block 2
        ├── Multi-Head Attention
        │   ├── Linear
        │   └── Linear
        ├── Layer Normalization
        └── Feed-Forward Network
            ├── Linear
            └── Activation
    └── ... (repeated Transformer Blocks)
```

There might also be a `LM Head` layer at the end to project hidden states back into the probability space of output IDs, depending on the downstream task.

**Modularity**

All these modules are implemented separately elsewhere and then assembled to form the `GPTModel`. Beyond `GPTModel` itself, these `Transformer Blocks` can be reused in other architectures like ViT, Llama, etc., perhaps with slight modifications to adapt to specific contexts.

**Hierarchy**

The `GPTModel` is the top-level module containing `Embedding`, `Positional Encoding`, and the collection of `Transformer Blocks`. 

- Each `Transformer Block` encapsulating `Multi-Head Attention`, `Layer Normalization`, and a `Feed-Forward Network`. 
  - Further, `Multi-Head Attention` and `Feed-Forward Network` are themselves composed of simpler `Linear` and `Activation` modules. 

> Note that the tree hierarchy does not imply sequential recursive execution of childeren modules. The exact computation logic is defined by user in forward function and may form a complex DAG.

> Yet, it's true that we register child module in the order they are executed by convention, and, most of the time, they are sequential.

---

## Layered System Design:

When working on the code this week, you may find that the autograd engine and tensor library hide the most complexitiy of underyling computation and backward tracing. Modules feel like a simple wrapper around the autograd system with state management.

Yes,that's exactly why we design it this way: seperating the system functionalities into distinct layers, where higher layer only relies on lower layers, and mostly, only its adjacent layer. This brings great similicity for both design and implement.

Module system is completely agnoistic to how autograd engine or tensor library work under the hood -- it just assumes they will do what they promise to do properly. Conversely, autograd engine or tensor library need not care how module system operates. 

> Though, from a designer's perspective, it is important to design good interfaces, with which higher layer can utilize lower layer efficiently and easily. This always requires a global view of the system.

Meanwhile this layered abstration is perfect in our problem, it is usually over-simplified or ideal in more complex systems. In those cases, engineers takes a middle ground between monolithic and layered design (i.e., modularity when possible). You will learn that in next year's operating system course.


## 📘 Additional Tutorials

Understanding how `nn.Module` works is fundamental to using (and then implementing) any modern deep learning library effectively. 

* [**`torch.nn.Module` Official Documentation**](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
  The definitive reference for all `nn.Module` functionality.

* [**Building Models with PyTorch**](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
  A beginner-friendly tutorial on how to define and use `nn.Module` to build networks.

* [**Saving and Loading Models in PyTorch**](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  A comprehensive guide on using `state_dict` to persist and restore model state.

---

# Code Guide

---

This week's focus leans more towards user-friendly system design rather than intricate low-level engineering. The elegance of a well-structured system lies in its simplicity for the end-user. So, to grasp what "user-friendly" means:

### **We highly recommend you getting familiar with PyTorch's module system before proceeding to code your own!**

Please refer to Addtional Tutorials above.

---

## Code Structure Overview

```bash
clownpiece
|-nn
| |- activations.py
| |- containers.py
| |- init.py
| |- layers.py
| |- loss.py
| |- module.py
|-...
```

- The `module.py` holds the core definition of abstract class `Module`, centralizing common functionalities for all modules.

- The `init.py` holds utility to initialize parameters in different probabilistic ways. 

- Other files contain concrete modules of a specific type suggested by the file name.

We'll follow these steps for implementation:

1. First, implement the core `Module` class in `module.py`, including
    - parameter and buffer management
    - state_dict save and load
    - \_\_repr\_\_ method to visualize module structure
2. Next, create several simplest concrete modules to rigorously test if the `Module`'s fundamental functionalities are correctly working.
3. Then, develop the `init.py` utilities for parameter initialization.
4. Finally, complete the implementation of other specific modules in `activations.py`, `layers.py`, `containers.py`, and `loss.py`.

5. Try out what you module system with two traditional DL tasks.

Due to the workload restriction and incompleteness of our autograd engine and tensor library, we can only explore a small portion of common modules.

> Anyway, a complete DL framework cannot be built from scratch in only weeks; don't be disappointed; We will build some interesting application with what we have!🤗

---

## Part 1: Core Module System

### Parameters/Buffers/Children Management
Starting by defining state storage for module, namely `Parameter` and `Buffer`.

They are trivial subclass of Tensor, with a preferred `requires_grad` value.

```python
class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)
```

Then, let's look at module's member variables:

```python
class Module(object):
  training: bool

  _parameters: Dict[str, Parameters]
  _buffers: Dict[str, Buffer]
  _modules: Dict[str, "Module"]
```

The `training` indicates whether the module is in training or inferencing mode. This affects the behavior of `forward`, for example:

- `Dropout` is disabled during inferencing.
- `BatchNorm` uses mean and var recorded from training as constant during inferencing

These three dictionaries record **immediate** parameters, buffers, and child modules belonging to this module. (but not from childrens).

You may have noticed that, in pytorch, when you assign a parameter `P` to attribute of module `M`, then `P` appears in `M`'s `parameters()` method without the need of any explicit declaration "I am a parameter, please register me".

This is accomplished by overriding `__setattr__(self, name: str, value: Any)` function: whenever `self.name = value` happens, this method is called. We can hijack the default assignment behavior, and detects, if `value` is instance of `Parameter`, `Buffer`, or `Module`, and register them to the corresponding dict. (if `value` is None of them, just call `object.__setattr__(self, name, value)`)

`__getattr__(self, name)` comes in couple with `__setattr__`, but it is called **only when** `name` is not found as ordinary attribute (i.e., `name` is hijecked at `__setattr__`).

Please complete:

```python
class Module(object):
  def __init__(self):
    # It's a good practice to add a mechanism to enforce that:
    #   All subclasses of Module must call super().__init__ in their __init__
    #   (User often forgets to do so!)
    # For example:
    #   add a boolean variable _init_called, 
    #   and check at beginning of __setattr__ call.
    #
    # this mechanism is optional and does not account for score.

    pass

  def train(self, flag: bool = True):
    # set module and submodule to training = flag
    pass

  def eval(self):
    # set module and submodule to inferencing mode
    pass

  def __setattr__(self, name, value):
    pass

  def __getattr__(self, name):
    pass
```

with these registry directories, it's easy to implement:

```python
class Module(object):

  """
    Parameter
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    # why does this function even exist? 
    # well, sometimes we want to register None as placeholder for disabled optioanl parameters. (e.g., bias in Linear)
    pass

  def parameters(self, recursive: bool=True) -> Iterable[Parameter]:
    # return a generator of all parameters in this module
    # yield immediate parameters first,
    # if recursive, then yield parameters from children.

    # HINT: use `yield` and `yield from` semantic
    pass

  def named_parameters(self, recursive: bool=True) -> Iterable[Tuple[str, Parameter]]:
    # similar to parameters, but return a name along with the parameter
    # the name is obtained by joining the recurisve attr name with '.'
    # for example
    """
      class A(Module):
        a: Parameter
        b: B

      class B(Moudle)
        c: Parameter
      
      Then, A.named_parameters() -> [
        ("a", ...),
        ("b.c", ...)
      ]
    """
    pass

  """
    Buffer
  """

  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    pass

  def buffers(self, recursive: bool=True) -> Iterable[Buffer]:
    pass

  def named_buffers(self, recursive: bool=True) -> Iterable[Tuple[str, Buffer]]:
    pass

  """
    Modules
  """

  def register_modules(self, module: Optional["Module"]):
    pass

  def modules(self, recursive: bool=True) -> Iterable["Module"]:
    pass

  def named_modules(self, recursive: bool=True) -> Iterable["Module"]:
    pass
```

---
### State Dict

As mentioned earlier, the module system manages state and therefore must provide a mechanism for saving and restoring this state from persistent storage. This capability is crucial for storing trained model weights and reloading them later—potentially on a different machine—for deployment or further use.

The model's state is exported using `state_dict()`, which returns a flat dictionary mapping names to tensors (`name -> Tensor`) (including parameters and buffers). The keys in this dictionary correspond to the names returned by the module’s `named_parameters()` and `named_buffers()` methods (joining attr name by '.').

The state dict of our first sample `SimpleNet` would be like:

```bash
{
  layer1.weight: Tensor(128, 784),
  layer1.bias: Tensor(128,),
  layer2.weight: Tensor(10, 128),
  layer2.bias: Tensor(10,),
}
```

Note that, when the tensor in state_dict is a shallow reference to real tensors. That's saying: no physcial copy at state_dict call.

We have implemented the pickle method (serialization) for Tensor class, so, you should be able to pickle the state dict to files.

The reverse operation is called `load_state_dict(state_dict)`: loads a dictionary from `state_dict()` call. The loading involves copying data from state_dict, not by assignment, since tensors in state_dict are shallow references.

The `load_state_dict` have a `strict` argument. When set, it checks that:

1. All the keys in state_dict are used. (i.e., no redundant weights)
2. All tensors' in state_dict matches the expected shape.
3. All expected keys are present in state_dict. (i.e., no missing weights)

Raise RuntimeError if any of the constraints is violated. 

Disabled optional weights should be present in state_dict, with `value = None`.

Please complete:

```python
class Module(object):
  def state_dict(self) -> Dict:
    pass

  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    pass
```

---

### Module Representation: `__repr__` and `extra_repr`

Every module should provide a clear and informative string representation, especially for debugging and visualization purposes. This is achieved through the `__repr__` method, which defines how the module is printed when you run `print(module)` or simply evaluate it in a REPL.

By default, the base `Module.__repr__()` method is implemented to recursively list all submodules along with their names and arguments. To customize how a module is represented (without changing the entire structure), subclasses should override the `extra_repr()` method instead.

#### `__repr__()`: Full Representation

This method automatically constructs a hierarchical string representation of the module and its children. It includes:

* The class name
* Any extra information provided by `extra_repr()`
* All submodules (indented for readability)

You **should not** override `__repr__` directly unless absolutely necessary. Instead, customize the output of `extra_repr()`.

To be specific, the output format can be detailed as:
```python
#basic:
ModuleClassName

#if extra_repr:
ModuleClassName(extra_repr)

#if has children:
ModuleClassName(
  (Child1Name): Child1Repr
  (Child2Name): Child2Repr
  ...
)

# add two space for indents for every recursion.
```

#### `extra_repr()`: Module-Specific Info

Override `extra_repr()` to include module-specific configuration like layer sizes, flags (e.g., `bias=True`), etc. This string is inserted inside the `__repr__` output right after the module name.

Example for a custom linear layer:

```python
class Linear(Module):
  def __init__(self, in_features, out_features, bias=True):
    ...

  def extra_repr(self) -> str:
    return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
```

When printed, this would result in:

```bash
Linear(in_features=784, out_features=128, bias=True)
```

This convention ensures a clean, standardized interface for module representation across all submodules in a deep learning model.

Recall the first example of `SimpleNet`, its \_\_repr\_\_ is:
```python
SimpleNet(
  (layer1): Linear(in_features=784, out_features=128, bias=True)
  (activation): ReLU()
  (layer2): Linear(in_features=128, out_features=10, bias=True)
)
```

Your implement for \_\_repr\_\_ will be awarded the credit as long as it's format is clear, properly indented and new-lined, and includes extra_repr, module class name, children name. It won't be judged by full-text matching.

Please complete:

```python
class Module(object):
  def __repr__(self) -> str:
    pass

  def extra_repr(self) -> str:
    return ""
```

---

## Forward

The `forward` for Module is an abstract method, and should be overridden by subclasses.

In PyTorch, user should use `__call__` method instead of `forward`, since it wraps in pre/post forward hooks. In our case, `__call__` and `forward` are just aliases.

```python
class Module(object):
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
```

---

## Part 2: Simplest Concrete Modules

**When implementing concrete modules, you should provide extra_repr whenever applicable. The convention is to include all arguments passed to `__init__`**.
<!-- Linear, Sigmoid -->

### Linear:

Linear layer captures an affine mapping: $y=x@W^T + b$. 

where:
-  $x$ is a (batch of) $n$-dim vectors, $y$ is a (batch of) $m$-dim vectors, $W^T$ is $(n,m)$-matrix (the weight), $b$ is $m$-dim vector (the bias).

- that saying, $n$ is the number of input channel, $m$ is the number of output channel.

- the $b$ is an optional parameter. If disabled, $y=x@W^T$.

> Why store $W$ in transposed way?  When processed in matmul, the right operand $W^T$ is to be tranposed back into $W$, which is contiguous!

```python
# layers.py
class Linear(Module):

  def __init__(self, in_features: int, out_features: int, bias: bool=True):
    # remember to wrap W and b in Parameter class, otherwise they won't be registered.
    # for now, init W, b with empty
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass

  def extra_repr(self):
    pass
```

### Tanh

Tanh is a simple wrapper around `class Tanh(Function)` to bring it into module system.

```python
# activations.py
class Tanh(Module):

  def __init__(self):
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass
```

Now, upon finishing this, you may run `grade_part1.py` to see if core module system is correct, and `grade_part2.py` to see if `Linear`, and `Tanh` are correct.

Please attach the output of `__repr__` test in `grade_part1.py` to your report, as it must be graded manually. (if you pass full text match test, for sure you will get the score, though.)

---

## Part 3: Init

Proper initialization of a model's parameters is a critical step that can significantly impact the training process. A good initialization strategy helps in several ways:
-   **Preventing Vanishing/Exploding Gradients**: If weights are too small, gradients can shrink to zero as they propagate backward through the network, halting learning. If they are too large, gradients can grow exponentially, leading to unstable training.
-   **Breaking Symmetry**: If all weights in a layer are initialized to the same value, all neurons in that layer will learn the same features. Proper initialization ensures that neurons start with different weights and can learn diverse representations.
-   **Speeding up Convergence**: A well-chosen initialization places the model in a "reasonable" starting state, often closer to a good solution, which can accelerate the convergence of the optimization algorithm.

In this step, you will implement several standard initialization schemes in `clownpiece/nn/init.py`. These functions will modify the input tensor *in-place* (hence the trailing underscore in their names) and should be performed **without tracking gradients**. (with `no_grad` context)


### Simple Initializations:

**Constant:**
```python
def constants_(tensor: Tensor, value: float):
  pass

def zeros_(tensor: Tensor):
  pass  

def ones_(tensor: Tensor):
  pass
```

**Normal:**
```python
def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
  pass
```

**Uniform:**
```python
def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0):
  pass
```

> Hint: you may find python's default random library helpful. Also try a helper function to initialize tensor's data in place with a generator and within no_grad context.


### Fan-in and Fan-out

Most modern initialization strategies scale the weights based on the number of input and output connections to a neuron, referred to as `fan_in` and `fan_out`. For a standard `Linear` layer with weight shape `(out_features, in_features)`:
-   `fan_in` is the number of input units, which is `in_features`. (`shape[-1]`)
-   `fan_out` is the number of output units, which is `out_features`. (`shape[-2]`)

### Gain

The statistical stability is of great importance during training, and one good rule is to initialize parameters in a way that 
- variance of input/output is consistent across layer. 

This is the primiary goal of many advanced initialzation function.

However, activation function complicates this as it distorts output and thus its variance. 

- A **ReLU** activation function sets all negative inputs to zero. For a symmetric input distribution centered at zero, this means half of the outputs become zero, which cuts the variance in half.
- A **tanh** function squashes its inputs into the range [-1, 1]. While its derivative is 1 at the origin (preserving variance for small inputs), it saturates for larger inputs, generally reducing variance.

An additional scalar at initialization to compentsate this is **gain**. Using $\text{gain}=g$ scales the variance by $g^2$

With some statistical assumptions, we can obtain the recommended gain value:

```python
import math
_gain_lookup_table = {
  "linear": 1.0,
  "idenity": 1.0,
  "sigmoid": 1.0,
  "tanh": 5/3,
  "relu": math.sqrt(2),
  "leaky_relu": lambda a: math.sqrt(2 / (1 + a * a)),
  "selu": 3/4
}

def calcuate_gain(nonlinearity: str, a: float = 0) -> float:
  nonlinearity = nonlinearity.lower()

  if nonlinearity not in _gain_lookup_table:
    raise KeyError(f"Unkown nonlinearity: {nonlinearity}, choices are {list(_gain_lookup_table.keys())}")
  
  value = _gain_lookup_table[nonlinearity]
  if nonlinearity == "leaky_relu":
    return value(a)
  else:
    return value

```


### Xavier (Glorot) Initialization

Proposed by Glorot and Bengio in 2010, Xavier initialization is designed to maintain the variance of activations and gradients across layers, particularly for saturating activation functions like `sigmoid` and `tanh`.

The goal is to set the variance of the weights `W` such that:
$$
\text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}
$$

You will implement two versions:

1.  **`xavier_uniform_(tensor, gain=1.0)`**: Samples from a uniform distribution $\mathcal{U}(-a, a)$, where:
    $$
    a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
    $$

2.  **`xavier_normal_(tensor, gain=1.0)`**: Samples from a normal distribution $\mathcal{N}(0, \sigma^2)$, where:
    $$
    \sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}
    $$

### Kaiming (He) Initialization

Proposed by He et al. in 2015, Kaiming initialization is specifically designed for layers followed by a Rectified Linear Unit (ReLU) or its variants. Since ReLU sets all negative inputs to zero, it effectively halves the variance of the activations. Kaiming initialization compensates for this.

The goal is to set the variance of the weights `W` to preserve the variance of the activations through the layer in the forward pass:
$$
\text{Var}(W) = \frac{2}{\text{fan\_in}}
$$

You will implement two versions:

1.  **`kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")`**: Samples from a uniform distribution $\mathcal{U}(-b, b)$, where:
    $$
    b = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    $$
    The `fan_mode` can be either "fan_in" or "fan_out".

    The `a` is the negative slope for leaky ReLU. (to be introduced later, for now, just use `calucalte_gain`)

2.  **`kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu")`**: Samples from a normal distribution $\mathcal{N}(0, \sigma^2)$, where:
    $$
    \sigma = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
    $$

Please complete the following functions in `init.py`. You can use python's default random library if it helps.

```python
# init.py

def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
  pass

def xavier_normal_(tensor: Tensor, gain: float = 1.0):
  pass

def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  pass

def kaiming_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  pass
```

After implementing these, you can use them to initialize the weights of your `Linear` layer. A common practice is to create a `reset_parameters` method inside your module to contain the initialization logic.

- initialize both weight and bias with $\text{uniform}(-b, b), b = \sqrt{\dfrac 1 {\text{fan\_in}}}$


```python
# Example usage in Linear.__init__
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(1 / self.in_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
          init.uniform_(self.bias, -bound, bound)
        # or equvialently, use 
        # init.kaiming_uniform_(self.weight, a = math.sqrt(5))
```

Run `grade_part3.py` to test your code. For advanced initialization methods (Xavier and Kaiming), we run each test 1000 times on 10×20 tensors, collecting all 200,000 data points to perform comprehensive statistical validation. We then directly compare the sample mean and variance/standard deviation against theoretical expected values using appropriate tolerance thresholds.

The statistical validation includes:
- **Mean validation**: Absolute deviation should be less than 0.1
- **Variance/Standard deviation validation**: Relative error should be less than 30-40%
- **Range validation**: For uniform distributions, all values should be within expected bounds

With 200,000 data points per test, the statistical estimates are highly reliable. It's very unlikely that your program will fail by chance if implemented correctly. The large sample size ensures robust detection of implementation errors while maintaining stability against random fluctuations.

---

## Part 4 Concrete Modules

## Activations

Besides `tanh`, which is a common activation function for recurrent neural networks, there are several other important activation functions you should know.

### Sigmoid
The **Sigmoid** function:
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$ 
squashes its input into the range (0, 1). It was historically popular but is now less common in hidden layers due to the vanishing gradient problem and its non-zero-centered output. It's primarily used in the output layer of a binary classifier to produce a probability.

### Tanh
The **Hyperbolic Tangent (tanh)** function:
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
 squashes its input into the range $(-1, 1)$. Its zero-centered output often makes it a better choice than sigmoid for hidden layers, but it still suffers from vanishing gradients for large inputs.

### ReLU (Rectified Linear Unit)

The **ReLU** function, 
$$
f(x) = \begin{cases}
  x & x \ge 0 \\
  0 & x < 0 
\end{cases}
$$

is the most widely used activation function in deep learning. It is computationally efficient and helps mitigate the vanishing gradient problem for positive inputs. However, it can suffer from the "Dying ReLU" problem, where neurons can become inactive and only output zero.

### Leaky ReLU
**Leaky ReLU** is a variant of ReLU that aims to solve the "Dying ReLU" problem. It is defined as 

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases}
$$

where $\alpha$ is a small positive constant (e.g., $0.01$). This allows a small, non-zero gradient when the unit is not active.

> Activations in ReLU family are not differentiable at $0$, but can be trivially assign either left or right derivative as a solution.

Please complete:

```python
class Sigmoid(Module):
  pass

class Tanh(Module):
  pass

class ReLU(Module):
  pass

class LeakyReLU(Module):
  pass
```

> Theoretically speaking, these operations should be supported by C++ backend. But I haven't decided which activations to include during week1, so ..., please write them as combination of elementary ops.

> Besides, you might want to add corresponding `Function` of these activations to `autograd/function.py` for a unified framework. (not mandatory)

---

## Containers

Containers are utilities to organize modules into groups. We will write three most common containers: `Sequential`, `ModuleList`, and `ModuleDict`.

### Sequential

`Sequential` is a container that chains modules together in the order they are passed to the constructor. The output of one module is fed as the input to the next. This is perfect for simple, feed-forward architectures like a basic multi-layer perceptron (MLP).

Example:
```python
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

output = model(input_tensor)
```
The `forward` method is implicitly defined to pass the input through each layer in sequence.

### ModuleList

`ModuleList` holds submodules in a list-like structure. Like a regular Python list, you can iterate over it, append to it, concatenate them, and access modules by index. However, unlike `Sequential`, it does **not** have a default `forward` method. User must define the `forward` pass themselves. 

The design purpose of `ModuleList` is that: if you put modules into a regular list, and assign it to a parent module, they won't be tracked as children! (a common mistake for beginner to module system). `ModuleList` solves this as itself is a `Module` subclass.


Example:
```python
class SequntialLinear(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleList(
          [Linear(10, 10) for _ in range(5)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### ModuleDict

`ModuleDict` holds submodules in a dictionary-like structure, allowing you to access them by string keys. Similar to `ModuleList`, it does **not** have a `forward` method. The design purpose of `ModuleDict` is the same as `ModuleList`.

Example:
```python
class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.choices = ModuleDict({
            '1': Linear(1, 10, 3),
            '2': Linear(10, 10)
        })
        self.activations = nn.ModuleDict([
            ['relu', ReLU()],
            ['selu', SELU()]
        ])


    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```

Please complete:
```python
# containers.py
class Sequential(Module):
  
  def __init__(self, *modules: Module):
    pass

  def forward(input):
    pass


class ModuleList(Module):
  
  def __init__(self, modules: Iterable[Module] = None):
    # hint: try to avoid using [] (which is mutable) as default argument. it may lead to unexpected behavor.
    # also be careful to passing dictionary or list around in function, which may be modified inside the function.
    pass

  def __add__(self, other: Iterable[Module]):
    pass

  def __setitem__(self, index: int, value: Module):
    pass

  def __getitem__(self, index: int) -> Module:
    pass

  def __delitem__(self, index: int):
    pass

  def __len__(self):
    pass

  def __iter__(self) -> Iterable[Module]:
    pass

  def append(self, module: Module):
    pass

  def extend(self, other: Iterable[Module]):
    pass

class ModuleDict(Module):
  
  def __init__(self, dict_: Dict[str, Module]):
    pass

  def __setitem__(self, name: str, value: Module):
    pass

  def __getitem__(self, name: str) -> Module:
    pass

  def __delitem__(self, name: str):
    pass

  def __len__(self):
    pass

  def __iter__(self) -> Iterable[str]:
    pass
  
  def keys(self) -> Iterable[str]:
    pass

  def values(self) -> Iterable[Module]:
    pass

  def items(self) -> Iterable[Tuple[str, Module]]:
    pass

  def update(self, dict_: Dict[str, Module]):
    pass
```

## Layers

Below are the main layers in `clownpiece/nn/layers.py`. Each is a subclass of `Module`, automatically registers parameters, supports `state_dict`, and applies proper weight initialization.

### Embedding
**Embedding** performs a lookup from an embedding matrix:

$$
  \text{out}[i] = \text{weight}[i]
$$

- Weight: shape `(num_embd, embd_dim)`, initialized with $\mathcal N(0,1)$.
- Forward input: integer tensor of indices of shape `(...)`
- Returns: tensor of shape `(..., embd_dim)`

You may think embedding as a linear layer, where input $i$ represent the $i$-th one-hot vector of num_embd dimensions. The weight matmuls to the on-hot vector and produce the output.

Examples: input is a batch of token ids, the output would be corresponding word vectors in the hidden space.



### LayerNorm
**LayerNorm** normalizes each sample over its last $d$ dimensions:
$$
  \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
where $\mu$ and $\sigma^2$ are the mean and variance computed over the normalized dimensions. Optional affine scale and shift are applied when `affine=True`. 

### BatchNorm
**BatchNorm** normalizes across a batch of $n$ samples:
$$
  \mu_B = \frac1m \sum_{i=1}^n x_i,
  \quad
  \sigma^2_B = \frac1m \sum_{i=1}^n (x_i - \mu_B)^2
$$

During training, running mean and var are estimated with momentum $m$:
$$
  \mu_{\text{running}} \leftarrow (1-m) \cdot \mu_{\text{runing}} + m \cdot \mu_B \\
  \sigma_{\text{running}}^2 \leftarrow (1-m) \cdot \sigma_{\text{running}}^2 + m \cdot \sigma_B^2
$$

and inputs are normalized with $\mu_B, \sigma_B$ as in LayerNorm. Optional affine scale and shift are applied when `affine=True`. 

During evaluation, stored running mean and variance are used instead of batch statistics.

### MultiheadAttention
**MultiheadAttention** implements scaled dot-product attention for multiple heads:

Single head attention:
$$
  \text{Attention}(Q,K,V) = \text{softmax}\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)\,V
$$

Multihead-attention splits the embedding into `num_heads` heads of dimension $d_k=\dfrac{\text{embed\_dim}}{\text{num\_heads}}$, applies attention in parallel, and concatenates the outputs back. 

A final linear projection combines the heads back to `embed_dim`.

> I admit that the introduction to layers here may be over-simplified. Please refer to torch document for more detail. 
> - [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
> - [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

Please complete:

```python

class Embedding(Module):
    def __init__(self, num_embd: int, embd_dim: int):
      pass

    def forward(self, x: Tensor) -> Tensor:
      pass
    
class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
      # input is reshaped to (-1, num_features) for normalziation.
      # for example:
      #   to normalize last two dimensions of tensor (batch_size, height, width)
      #   then num_features should be height x width
      # this interface differs from pytorch
      pass

    def forward(self, x: Tensor) -> Tensor:
      pass

class BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
      pass

    def forward(self, x: Tensor) -> Tensor:
      pass
    
class MultiheadAttention(Module):
    def __init__(self, hidden_dim: int, num_heads: int, bias: bool = True):
      pass

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None):
      pass
```

---

## Loss

Loss functions are a critical component of the training process, quantifying the difference—or "error"—between a model's predictions and the ground truth labels. The goal of training is to adjust the model's parameters to minimize the value returned by the loss function.

### MSELoss (Mean Squared Error)

**Mean Squared Error** is the standard loss function for **regression tasks**, where the goal is to predict a continuous value. It measures the average of the squares of the errors between the predicted and actual values.

The formula for a batch of $N$ samples is:
$$
\mathcal{L}(y_{\text{pred}}, y_{\text{true}}) = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}, i} - y_{\text{pred}, i})^2
$$
-   $y_{\text{true}}$ is the tensor of ground truth values.
-   $y_{\text{pred}}$ is the tensor of model predictions.

Squaring the difference penalizes larger errors more heavily and ensures the result is always non-negative. 

Also, MSE loss is perhas the most intuitive differentiable non-negative loss.

### CrossEntropyLoss

**Cross-Entropy Loss** is the standard loss function for **classification tasks**. It measures the dissimilarity between the predicted probability distribution and the true distribution (which is typically a one-hot vector where the correct class has a probability of 1 and all others have 0).

For a single data point, the cross-entropy loss is calculated as:
$$
\mathcal{L}(y_{\text{pred}}, y_{\text{true}}) = - \sum_{c=1}^{C} y_{\text{true}, c} \cdot \log(y_{\text{pred}, c})
$$
-   $C$ is the number of classes.
-   $y_{\text{true}, c}$ is 1 if $c$ is the correct class, and 0 otherwise.
-   $y_{\text{pred}, c}$ is the model's predicted probability for class $c$.

This equation simplifies to $-\log(p_k)$, where $p_k$ is the predicted probability of the correct class $k$. Minimizing this loss is equivalent to maximizing the log-probability of the correct class.

> **Numerical Stability**: In common practice, `CrossEntropyLoss` is combined with a `LogSoftmax` function. This operation is more numerically stable than applying a `Softmax` and then a `log` separately by avoiding `log(0)`. User only feed logits (unnormalized output from classification model) and correct label to cross-entropy. 
> A workaround in our case is to replace:
> ```python
> log_probs = logits.softmax(dim=-1).log()
> ```
> with
> ```python
> log_probs = logits - logits.exp().sum(dim=-1, keepdims=True).log()
> ```


Please complete:
```python
# loss.py
class MSELoss(Module):
  def __init__(self, reduction: str = 'mean'):
    pass
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    pass

class CrossEntropyLoss(Module):
  def __init__(self, reduction: str = 'mean'):
    pass
  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    # logits is of shape (..., num_class)
    # target is of shape (...), and it's value indicate the index of correct label

    # You need to ensure your implement is differentiable under our autograd engine.
    # However, you can assume target has requires_grad=False and ignore the gradient flow towards it.
    pass
```

Run `grade_part4_{category of module}.py` to test your code, or `grade_part4.py` to test all.

----

# Putting It All Together

Congratulations! You have now built a complete, albeit miniature, core of a deep learning framework. What's left for next week are peripheral utilities that make the system more adaptive and easy to use in various scenarios (CV, Audio, etc.).

For now, let's put your framework to the test with two classic machine learning tasks that don't require complex data loaders or pre-processing pipelines.

---

## 1. Predict Real Estate Value (Regression)

**Task**: Predict the price of a house based on several features. This is a typical **regression** problem, where the goal is to predict a continuous output value.

Please complete the `tests/week3/estate_value_predict/main.ipynb`.

---

## 2. Classify Handwritten Digits (Classification)

**Task**: Identify the digit (0 through 9) from a 28x28 pixel image of a handwritten number. This is a **classification** problem, where the goal is to predict a discrete category.

Please complete the `tests/week3/mnist/main.ipynb`.

---

# Optional Challange: Add Support for Conv2d

~~**Due to TAs' design failure**~~ due to incomplete feature of our tensor library and autograd engine, we are unable to support a important class of Module: **Convolution**, which is dominant in previous era of DL (right before transformers).

Your task is to add support for `Conv2D` in our system. This involves adding basic operations at C++ backend, binding in Python, Functions in autograd, and finally, a new `Conv2D` in module system.

For what `Conv2D` is, refer to [this link](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

These two basic operations may be of your interests:

- **Unfold**: also called `im2col`, which extends 2D image into `(num_patches, patch_dim)` given a specific kernel size. After this operation, Conv2D can be done by simply matmul with kernel. [link](https://pytorch.org/docs/stable/generated/torch.nn.Fold.html)

```python
  def Conv2D(images, kernel_size, kernel_weight):
    unfolded = unfold(images, kernel_size)
    return matmul(unfolded, kernel_weight)
```

- **Fold**: also called `col2im`, which reduces the results from unfold, summation over overlapped area, and average them. (but not strictly the reverse of unfold). You need this function for backward of unfold. [link](https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html)


Besides, **Padding** may also be necessary.

### Your Task:

- Implement a `Conv2d` of the simplist case: `stride=1`, `padding='same'`, `dilation=1`,`groups=1`.
  - naive solution of "for loops kernel" in python is not allowed
  - naive solution of "for loops kernel" in C++ is allowed, but only partial credit
  - efficient solution using "unfold" is encouraged.

```python
class Conv2D(Module):

  def __init__(self, in_channels, out_channels, kernel_size_height, kernel_size_width):
    pass

  def forward(self, x):
    pass
```

- Include a section in your report to clearly explain:
  - How do you implement `Conv2d`?

  - What's the exact semantic of `unfold` and it's backward? How can it be utilized in `Conv2d`?
    - TA fAKe was occupied with 仕事, and cannot understand by himself in limited free time
    - Even if you fail to give a working `Conv2D` implement, by explaining its principle clearly, you will receive partial credit.

---


## Submit Your Homework

First, make sure that you passed the `grade_all.py`.

Then, you should write a detailed **report** under `docs/week3`, to describe the challenges you have encountered, how did you solved them, and what are the takeaways. (Also, attach the grader summary part from output of `grade_all.py`). This report accounts for part of your score.

There is no optional challenge for week2, but if you complete week1'
s OC, you may include it in this week's report.

Finally zip the entire project folder into lab-week3.zip, and submit to canvas.

Make sure that the TAs can run the `grade_all.py` and find your report from your submission.