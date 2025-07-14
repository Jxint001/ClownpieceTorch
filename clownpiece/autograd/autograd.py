from typing import Dict, Iterable, List, Optional, Union, Any

#from clownpiece.autograd.function import AccumulateGrad
from clownpiece.tensor import Tensor, ones_like, zeros_like
from clownpiece.utils_ import wrap_tuple

import queue
from collections import deque
from threading import Thread, Lock

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
        if not isinstance(tensor, Tensor):
            return Edge(-1, None)
        
        if tensor.requires_grad == True:
            if tensor.grad_fn is not None:
                # not a leaf
                return Edge(tensor.output_nr, tensor.grad_fn)
            else:
                # an input tensor and requires grad (leaf also)
                from clownpiece.autograd.function import AccumulateGrad
                return Edge(0, AccumulateGrad(tensor))
        else: 
            # leaf that requires no grad
            return Edge(-1, None)

class GraphRoot(Node):
    """
    Root node in the computation graph.
    """

    def __init__(self, tensor: Tensor, grad: Tensor):
        super().__init__()
        self.output_nr = 0
        # step1. store the grad
        self.root_grad = grad
        # step2. create a single edge points to tensor.grad_fn
        self.next_edges.append(Edge(0, tensor.grad_fn))
    
    def run(self, *args, **kargs):
        # step1. return the stored grad
        return self.root_grad

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
        # step1. run the node with inputs
        outputs = self.node.run(*self.inputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = wrap_tuple(outputs)

        # step2. fill the input buffer in GraphTask
        for e, grad in zip(self.node.next_edges, outputs):
            if e.node is not None and grad is not None:
                self.base.fill_input(e.node, grad, e.input_nr)


class GraphTask():
    """
    GraphTask wraps the execution of a computation graph.
    """
    
    roots: List[Node] # GraphRoots instances
    nodes: List[Node] # all nodes in the computation graph
    dependencies: Dict[Node, int] # count of inbound degree for topological sort
    inputs_buffer: Dict[Node, List[Tensor]] # inputs_buffer to accumulate intermediate results.
    
    def print_graph(self):
        print("\n","graph:")
        print("##########################")
        print("there are %d nodes in the graph, %d of which are roots" %(self.nodes.__len__(), self.roots.__len__()))
        print("dependencies:")
        node = list(self.dependencies.keys())
        deps = list(self.dependencies.values())
        n = node.__len__()
        for i in range(n):
            print("node %d has %d dependencies" % (node[i].node_id, deps[i]))
        print("inputs_buffer:")
        bufs = list(self.inputs_buffer.keys())
        lsts = list(self.inputs_buffer.values())
        m = bufs.__len__()
        for i in range(m):
            print("node %d has buffer: " % bufs[i].node_id, lsts[i])
        
        print("node_id and corresponding function")
        for nd in self.nodes:
            print(nd.node_id, nd)
        print("##########################\n")

    def __init__(self, roots: List[Node]):
        roots = wrap_tuple(roots)
        roots = [root for root in roots if root is not None]
        
        if not roots:
            raise ValueError("roots is empty")
    
        self.lock = Lock()
        self.roots = roots
        self.nodes = []
        self.dependencies = {}
        self.inputs_buffer = {}
        self._construct_graph()
        # self.print_graph()  # for debug
        
    # helper function to assign node_id and initialize self.nodes, dependencies and inputs_buffer
    def _construct_graph(self):
        que = queue.Queue()
        for rt in self.roots:
            que.put(rt)

        id = 0
        for ns in self.roots:
            ns.node_id = id
            id += 1
        self.nodes.extend(self.roots)
        self.dependencies = {key : 0 for key in self.roots}  # roots have no dependency
        self.inputs_buffer = {key : [] for key in self.roots}  # roots have nothing to accumulate


        while not que.empty():
            node = que.get()
            if node not in self.inputs_buffer:
                self.inputs_buffer[node] = []

            for e in node.next_edges:
                nxt = e.node
                if nxt is None:
                    continue

                if nxt not in self.dependencies:  # new node in graph

                    que.put(nxt)
                    nxt.node_id = id
                    id += 1
                    self.nodes.append(nxt)
                    self.dependencies[nxt] = 0
                self.dependencies[nxt] += 1


        for nd in self.nodes:
            self.inputs_buffer[nd] = [None] * self.dependencies[nd]

        
    # execute
    def run(self):
        # self._run_single_thread()
        self._run_multi_thread()

    # for debug
    def _run_single_thread(self):
        # perform topological sort to execute the graph
        dq = deque([NodeTask(node, self.inputs_buffer[node], self) for node in self.roots])
        while dq:
            node_task = dq.popleft()
            node_task.run()

            for e in node_task.node.next_edges:
                nxt = e.node
                if nxt is not None:
                    self.dependencies[nxt] -= 1
                    if self.dependencies[nxt] == 0:
                        dq.append(NodeTask(nxt, self.inputs_buffer[nxt], self))
                        self.inputs_buffer.pop(nxt)

    # for production
    def _run_multi_thread(self):
        # step1. maintain a shared ready queue for NodeTasks
        sh_q = queue.Queue()
        finished = set()

        for node in self.roots:
            sh_q.put(NodeTask(node, self.inputs_buffer[node], self))
        # step2. def a worker function, similar to _run_single_thread.
        # be careful: do not use `while queue is not empty` as exit condition directly. (why?)
        def worker():
            while True:
                try:
                    node_task = sh_q.get_nowait()
                except queue.Empty:
                    # check whether all node_task are finished
                    with self.lock:
                        if len(finished) == len(self.nodes):
                            break
                        else:
                            continue
                node_task.run()
                with self.lock:
                    finished.add(node_task.node)
                for e in node_task.node.next_edges:
                    nxt = e.node
                    if nxt is not None:
                        with self.lock:
                            self.dependencies[nxt] -= 1
                            if self.dependencies[nxt] == 0:
                                sh_q.put(NodeTask(nxt, self.inputs_buffer[nxt], self))
                                self.inputs_buffer.pop(nxt)
                sh_q.task_done()
    
        # step3. spawn multiple worker threads.
        num_thread = 2
        threads = [Thread(target = worker, args = (),  kwargs={}) for _ in range(num_thread)]
        for t in threads:  t.start()
        # step4. wait for threads to join.
        for t in threads:  t.join()
                    
    # accumulate input_grad to self.inputs_buffer[node][input_nr]
    def fill_input(self, node: Node, input_grad: Tensor, input_nr: int):
        with self.lock:
            if self.inputs_buffer[node][input_nr] is None:
                self.inputs_buffer[node][input_nr] = input_grad
            else:
                self.inputs_buffer[node][input_nr] += input_grad


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