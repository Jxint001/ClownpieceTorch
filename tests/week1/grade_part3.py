import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="is_contiguous1: True case", score=10)

def is_contiguous1(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    a = impl.Tensor(lst)
    return a.is_contiguous()

# @testcase(name="is_contiguous2: False case", score=10)

@testcase(name="simple_index1: Valid operations", score=10)

def simple_index1(impl = torch):
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = impl.Tensor(lst)
    b = a.clone()
    b[0] = 100
    # import pickle
    # p = a[1:5]
    # print("Original Tensor=", p)
    # s = pickle.dumps(p)
    # print("Serialized=", s)
    # t = pickle.loads(s)
    # # return a[1]
    return (a[1], a[4], a[5], b)

@testcase(name="simple_index2: Valid operations", score=10)

def simple_index2(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    a = impl.Tensor(lst)
    b = a.clone()
    b[0][0] = 100
    return (a[0][0], a[1][1], a[0][1], a[1][2], 
            a[0][3], a[1][3], a[0][4], a[1][4], b)
    
@testcase(name="simple_index3: Invalid operations", score=10)

def simple_index3(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    a = impl.Tensor(lst)
    
    if impl.__name__ == "torch":
        return True
    
    else:
        try:
            return a[0][0][0]
        except RuntimeError as e:
            try:
                return a[0][6]
            except RuntimeError as e:
                return True
            
@testcase(name="simple_index4: Slice of data", score=10)

def simple_index4(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    a = impl.Tensor(lst)
    b = a.clone()
    b[0] = impl.Tensor([10, 20, 30, 40, 50])
    return (a[0], a[1], a[2], a[3], a[4], b[0])
            
@testcase(name="simple_slice1: 1D tensor", score=10)

def simple_slice1(impl = torch):
    # slice_t is a <int, int> pair, so operations like [1:8:2] is not supported
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = impl.Tensor(lst)
    return (a[1:5], a[2:8], a[0:-1], a[2:-2], a[3:], a[:5], a[1:-8], a[1:-100])

@testcase(name="simple_slice2: 2D tensor", score=10)

def simple_slice2(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    a = impl.Tensor(lst)
    return (a[1:3][2:4], a[0:2][1:3], a[1:4][0:2], 
            a[2:5][1:4], a[0:5][0:5], a[1:3][1:5], 
            a[2:4][0:-1], a[1:-2][1:-3], a[:][0])
    
@testcase(name="complex_slice: 2D tensor", score=10)

def complex_slice(impl = torch):
    
    lst =[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    lst2 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    a = impl.Tensor(lst)
    b = impl.Tensor(lst2)
    ret1 = a[0, 1:3]
    ret2 = a[:, 2:5]
    ret3 = b[:, 0]
    
    print(ret3)

    return ret1, ret2, ret3

@testcase(name="is_contiguous2: True case", score=10)

def is_contiguous2(impl = torch):
    lst = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    a = impl.Tensor(lst)
    b = a[0:2][1:3]
    return b.is_contiguous()


def testsets_part3():
    is_contiguous1()
    
    simple_index1()
    simple_index2()
    simple_index3()
    simple_index4()
    
    simple_slice1()
    simple_slice2()
    complex_slice()
    
    is_contiguous2()

if __name__ == "__main__":
  print("Beginning grading part 3")
#   set_debug_mode(True)
  testsets_part3()
  grader_summary("part3")