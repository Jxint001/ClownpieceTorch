from typing import Dict, Iterable, List, Optional, Union, Any

from clownpiece.tensor import Tensor, ones_like, zeros_like
from clownpiece.utils_ import wrap_tuple

"""
    Autograd Module
"""

# autograd/autograd.py
class Node():
    node_id: int
    next_edges: List["Edge"]

    def __init__(self):
        self.node_id = None
        self.next_edges = []
        
    def run(self, *args, **kargs):
        raise NotImplementedError("run method not implemented for abstract Node instance")
    
    # define __hash__ and __eq__ to use Node as dict's key
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id

class Edge():

    input_nr: int # the Edge points to the i-th input of target Node
    node: Optional[Node] # target node the Edge points to

    def __init__(self, input_nr: int, node: Optional[Node]):
        self.input_nr = input_nr
        self.node = node
    
    @staticmethod
    def gradient_edge(tensor: Tensor) -> "Edge":
      # your implement here

      # case 1: tensor is not a leaf tensor -> use it's grad_fn and output_nr

      # case 2: tensor is a leaf tensor and requires grad -> AccumulateGrad Function

      # case 3: tensor is a leaf tensor and requires no grad -> node = None
      pass

class GraphRoot(Node):
    """
    Root node in the computation graph.
    """

    def __init__(self, tensor: Tensor, grad: Tensor):
      # your implement here

      # step1. store the grad
      # step2. create a single edge points to tensor.grad_fn
      pass
    
    def run(self, *args, **kargs):
      # your implement here

      # step1. return the stored grad
      pass

class NodeTask():
    """
    NodeTask wraps a Node and all its input. 
    It's a ready-to-run Node in GraphTask.
    """

    base: "GraphTask"
    node: Node
    inputs: List[Tensor]
    
    def __init__(self, node: Node, inputs: List[Tensor], base: "GraphTask"):
        self.base = base
        self.node = node
        self.inputs = inputs
        
    def run(self):
        # your implement here

        # step1. run the node with inputs

        # step2. fill the input buffer in GraphTask
        pass


class GraphTask():
    
    """
    GraphTask wraps the execution of a computation graph.
    """
    
    roots: List[Node] # GraphRoots instances
    nodes: List[Node] # all nodes in the computation graph
    dependencies: Dict[Node, int] # count of inbound degree for topological sort
    inputs_buffer: Dict[Node, List[Tensor]] # inputs_buffer to accumulate intermediate results.
    
    def __init__(self, roots: List[Node]):
        roots = wrap_tuple(roots)
        roots = [root for root in roots if root is not None]
        
        if not roots:
            raise ValueError("roots is empty")
    
        self.roots = roots
        self.nodes = []
        self.dependencies = {}
        self.inputs_buffer = {}
        self._construct_graph()
        
    # helper function to assign node_id and initialize self.nodes, dependencies and inputs_buffer
    def _construct_graph(self):
        # your implement here
        pass
        
    # execute
    def run(self):
        # your implement here
        pass

    # for debug
    def _run_single_thread(self):
        # your implement here

        # perform topological sort to execute the graph

        # while queue is not empty:
        # 1. node_task = queue.pop()
        # 2. node_task.run()
        # 3. decrement dependencies count for target nodes of outbound edges
        # 4. enqueue a new NodeTask if dependencies drops to zero. (remember to delete the node in inputs_buffer to release memory.)
        pass

    # for production
    def _run_multi_thread(self):
        # your implement here

        # step1. maintain a shared ready queue for NodeTasks

        # step2. def a worker function, similar to _run_single_thread.
        # be careful: do not use `while queue is not empty` as exit condition directly. (why?)

        # step3. spawn multiple worker threads.

        # step4. wait for threads to join.
        pass
                    
    # accumulate input_grad to self.inputs_buffer[node][input_nr]
    def fill_input(self, node: Node, input_grad: Tensor, input_nr: int):
        # your implement here

        pass


"""
    Execute backward pass.    
"""
def backward(tensors: Union[Tensor, List[Tensor]], grads: Optional[Union[Tensor, List[Tensor]]] = None):
    tensors = wrap_tuple(tensors)

    if grads is None:
        grads = [ones_like(tensor) for tensor in tensors]
    grads = wrap_tuple(grads)
    
    # wrap with GraphRoots
    graph_roots = [
        GraphRoot(tensor, grad) for tensor, grad in zip(tensors, grads) if tensor.requires_grad
    ]

    # execute with GraphTask
    gt = GraphTask(graph_roots)
    gt.run()