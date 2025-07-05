"""
    Shape Manipulation Functions
"""

import torch
from graderlib import set_debug_mode, testcase, grader_summary, print_separate_line

# Test data
test_data_basic = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # For reshape, view, etc.
test_data_chunk_split = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]  # For chunk split 3 * 4
test_data_sequeeze = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]  # For sequeeze
test_data_stack = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]  # For stack
test_data_cat = [[[1.0, 2.0]], [[3.0, 4.0]]]  # For cat
test_data_broadcast = [[1.0], [2.0]]  # For broadcast_to

@testcase(name="permute_op", score=10)
def permute_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    perm = [1, 0]  # Swap axes
    b = a.permute(perm)
    b.backward(b)
    return a.grad, b

@testcase(name="transpose_op", score=10)
def transpose_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    dim0, dim1 = 0, 1
    b = a.transpose(dim0, dim1)
    b.backward(b)
    return a.grad, b

@testcase(name="reshape_op", score=10)
def reshape_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    shape = [3, 2]
    b = a.reshape(shape)
    b.backward(b)
    return a.grad, b

@testcase(name="view_op", score=10)
def view_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    shape = [3, 2]
    b = a.view(shape)
    b.backward(b)
    return a.grad, b

@testcase(name="narrow_op", score=10)
def narrow_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    dim, start, length = 1, 0, 2
    b = a.narrow(dim, start, length)
    b.backward(b)
    return a.grad, b

@testcase(name="chunk_op", score=10)
def chunk_op(impl=torch):
    a = impl.Tensor(test_data_chunk_split)
    a.requires_grad_()
    chunks = 2
    outputs = a.chunk(chunks, dim=1)
    b = sum(outputs)
    b.backward(impl.ones_like(b))
    
    c = impl.Tensor(test_data_chunk_split) 
    c.requires_grad_()
    chunks = 2
    outputs = c.chunk(chunks, dim=0)
    d = sum(outputs)
    d.backward(impl.ones_like(d))
    
    return a.grad, b, c.grad, d

@testcase(name="split_op", score=10)
def split_op(impl=torch):
    a = impl.Tensor(test_data_chunk_split)
    a.requires_grad_()
    split_sizes = [2, 2]
    outputs = a.split(split_sizes, dim=1)
    b = sum(outputs)
    b.backward(impl.ones_like(b))
    
    c = impl.Tensor(test_data_chunk_split)
    c.requires_grad_()
    split_sizes = [1, 2]
    outputs = c.split(split_sizes, dim=0)
    d = sum(outputs)
    d.backward(impl.ones_like(d))
    return a.grad, d, c.grad, d

@testcase(name="stack_op", score=10)
def stack_op(impl=torch):
    tensors = [impl.Tensor(data) for data in test_data_stack]
    for t in tensors:
        t.requires_grad_()
    b = impl.stack(tensors, dim=0)
    b.backward(impl.ones_like(b))
    grads = [t.grad for t in tensors]
    return grads + [b]

@testcase(name="cat_op", score=10)
def cat_op(impl=torch):
    tensors = [impl.Tensor(data) for data in test_data_cat]
    for t in tensors:
        t.requires_grad_()
    b = impl.cat(tensors, dim=0)
    b.backward(impl.ones_like(b))
    grads = [t.grad for t in tensors]
    return grads + [b]

@testcase(name="squeeze_op", score=10)
def squeeze_op(impl=torch):
    a = impl.Tensor(test_data_sequeeze)
    a.requires_grad_()
    dim = 0
    b = a.squeeze(dim)
    b.backward(impl.ones_like(b))
    return a.grad, b

@testcase(name="unsqueeze_op", score=10)
def unsqueeze_op(impl=torch):
    a = impl.Tensor(test_data_basic)
    a.requires_grad_()
    dim = 0
    b = a.unsqueeze(dim)
    b.backward(impl.ones_like(b))
    return a.grad, b

@testcase(name="broadcast_to_op", score=10)
def broadcast_to_op(impl=torch):
    a = impl.Tensor(test_data_broadcast)
    a.requires_grad_()
    shape = [2, 2]
    b = a.broadcast_to(shape)
    b.backward(impl.ones_like(b))
    
    shape = [3, 2, 2]
    c = a.broadcast_to(shape)
    c.backward(impl.ones_like(c))
    
    return a.grad, b, c

@testcase(name="broadcast", score=10)
def broadcast_op(impl=torch):
    a = impl.Tensor(test_data_broadcast)
    a.requires_grad_()
    b = impl.Tensor([1.0])
    b.requires_grad_()
    if impl == torch:
        c = impl.broadcast_tensors(a, b)
    else:
        c = impl.broadcast((a, b))
    d = c[0] + c[1]
    d.backward(impl.ones_like(d))
    
    return a.grad, b.grad, d

def testsets_part6():
    print_separate_line()
    print("Testing Part6 Shape Manipulation Functions...")
    permute_op()
    transpose_op()
    reshape_op()
    view_op()
    narrow_op()
    chunk_op()
    split_op()
    stack_op()
    cat_op()
    squeeze_op()
    unsqueeze_op()
    broadcast_to_op()
    broadcast_op()

if __name__ == "__main__":
    set_debug_mode(True)
    testsets_part6()
    grader_summary("Part6")