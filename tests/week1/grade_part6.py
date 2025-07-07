import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="sum1: sum along dim=0", score=10)

def sum1(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 0)

@testcase(name="sum2: sum along dim=1", score=10)

def sum2(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 1)

@testcase(name="sum3: sum with keepdims", score=10)

def sum3(impl=torch):
    a = impl.Tensor([[1, 2, 3], [4, 5, 6]])
    return a.sum(dim = 1, keepdims = True)

@testcase(name="sum4: sum for higher dimension", score=10)

def sum4(impl=torch):
    a = impl.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    return a.sum(dim = 1, keepdims = True)

@testcase(name="max1: max along dim=0", score=10)

def max1(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 0)
    return (vals, idxs)

@testcase(name="max2: max along dim=1", score=10)

def max2(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 1)
    return (vals, idxs)

@testcase(name="max3: max with keepdims", score=10)

def max3(impl=torch):
    a = impl.Tensor([[1, 5, 2], [4, 3, 6]])
    vals, idxs = a.max(dim = 1, keepdims = True)
    return (vals, idxs)

@testcase(name="max4: max for higher dimension", score=10)

def max4(impl=torch):
    a = impl.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    return a.max(dim = 1)

@testcase(name="softmax1: softmax for tensor", score=10)

def softmax1(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [2., 4., 6.]])
    return a.softmax(dim = 0)

@testcase(name="softmax2: softmax for bigger tensor", score=10)

def softmax2(impl=torch):
    a = impl.Tensor([[[1., -2., 3., 4., 5.], [6., 7., 8., 9., -10.]],
                     [[11., 12., 13., 14., 15.], [16., 17., 18., 19., -20.]],
                     [[21., 22., 23., 24., 25.], [26., -27., 28., 29., -30.]],
                     [[31., 32., -33., 34., 35.], [36., 37., 38., 39., -40.]]])
    return a.softmax(dim = 1)

@testcase(name="mean: mean for 1D tensor", score=0)

def mean1(impl=torch):
    a = impl.Tensor([1., 2., 3., 4., 5.])
    return a.mean(dim=0)

@testcase(name="mean: mean for 2D tensor", score=0)

def mean2(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [4., 5., 6.]])
    return a.mean(dim=0), a.mean(dim=1)

@testcase(name="mean: mean with keepdims", score=0)

def mean3(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [4., 5., 6.]])
    return a.mean(dim=0, keepdims=True), a.mean(dim=1, keepdims=True)

@testcase(name="var: variance for 1D tensor", score=0)

def var1(impl=torch):
    a = impl.Tensor([1., 2., 3., 4., 5.])
    return a.var(dim=0, unbiased=False)

@testcase(name="var: variance for 2D tensor", score=0)

def var2(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [5., 5., 6.]])
    return a.var(dim=0, unbiased=False), a.var(dim=1, unbiased=False)

@testcase(name="var: variance with keepdims", score=0)

def var3(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [5., 5., 6.]])
    return a.var(dim=0, keepdims=True, unbiased=False), a.var(dim=1, keepdims=True, unbiased=False)

@testcase(name="var: variance unbiased", score=0)

def var4(impl=torch):
    a = impl.Tensor([[1., 2., 3.], [5., 5., 6.]])
    return a.var(dim=0, unbiased=True), a.var(dim=1, unbiased=True)

def testsets_part6():
    sum1()
    sum2()
    sum3()
    sum4()
    max1()
    max2()
    max3()
    max4()
    softmax1()
    softmax2()
    
    mean1()
    mean2()
    mean3()
    var1()
    var2()
    var3()
    var4()

if __name__ == "__main__":
    print("Beginning grading part6")
    # set_debug_mode(True)
    testsets_part6()
    grader_summary("part6")