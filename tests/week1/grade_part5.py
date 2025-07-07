import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="matmul1: 1D @ 1D = scalar", score=10)

def matmul1(impl=torch):
    a = impl.Tensor([1, 2, 3])
    b = impl.Tensor([4, 5, 6])
    result = a @ b
    return result.item()

@testcase(name="matmul2: 1D @ 2D = 1D", score=10)

def matmul2(impl=torch):
    a = impl.Tensor([1, 2])
    b = impl.Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2,3)
    result = a @ b  # (2,) @ (2,3) -> (3,)
    return result

@testcase(name="matmul3: 2D @ 1D = 1D", score=10)

def matmul3(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
    b = impl.Tensor([7, 8, 9])  # (3,)
    result = a @ b  # (2,3) @ (3,) -> (2,)
    return result

@testcase(name="matmul4: 2D @ 2D = 2D", score=10)

def matmul4(impl=torch):
    a = impl.Tensor([[1, 2], [3, 4]])  # (2,2)
    b = impl.Tensor([[5, 6], [7, 8]])  # (2,2)
    result = a @ b
    return result

@testcase(name="matmul5: nD @ nD with broadcasting", score=10)

def matmul5(impl=torch):
    # a: (2, 1, 3, 4), b: (1, 5, 4, 6)
    a = impl.zeros((2, 1, 3, 4))
    b = impl.zeros((1, 5, 4, 6))
    # Output: (2, 5, 3, 6)
    c = a @ b
    return c.shape

@testcase(name="matmul6: Batch matmul with broadcasting", score=10)

def matmul6(impl=torch):
    # (batch, n, m) @ (batch, m, p)
    a = impl.zeros((3, 2, 4))
    b = impl.zeros((1, 4, 5))
    c = a @ b  # (3,2,4) @ (1,4,5) -> (3,2,5)
    return c.shape

@testcase(name="matmul7: shape mismatch throws", score=10)

def matmul7(impl=torch):
    a = impl.ones((2, 3))
    b = impl.ones((4, 5))
    
    if impl.__name__ == "torch":
        return True
    
    else:
        try:
            c = a @ b
            return False
        except RuntimeError:
            pass

        try:
            a = impl.ones((2, 3, 4))
            b = impl.ones((2, 3, 5))
            c = a @ b
            return False
        except RuntimeError:
            return True

@testcase(name="matmul8: broadcasting error throws", score=10)

def matmul8(impl=torch):
    a = impl.ones((2, 3, 4))
    b = impl.ones((3, 4, 5))
    
    if impl.__name__ == "torch":
        return True
    
    else:
        try:
            c = a @ b  # batch dim 2 vs 3, cannot broadcast
            return False
        except RuntimeError:
            return True

def testsets_part5():
    matmul1()
    sys.stdout.flush()
    matmul2()
    sys.stdout.flush()
    matmul3()
    sys.stdout.flush()
    matmul4()
    sys.stdout.flush()
    matmul5()
    sys.stdout.flush()
    matmul6()
    sys.stdout.flush()
    matmul7()
    sys.stdout.flush()
    matmul8()
    sys.stdout.flush()

if __name__ == "__main__":
    print("Beginning grading part5")
    # set_debug_mode(True)
    testsets_part5()
    grader_summary("part5")