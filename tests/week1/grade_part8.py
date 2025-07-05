import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="ones: shape=[2,3]", score=10)

def ones1(impl):
    t = impl.ones([2, 3])
    return t

@testcase(name="ones_like: shape=[2,2]", score=10)

def ones_like1(impl):
    ref = impl.zeros([2, 2])
    t = impl.ones_like(ref)
    return t

@testcase(name="ones_like: scalar", score=10)

def ones_like_scalar(impl):
    ref = impl.zeros([])
    t = impl.ones_like(ref)
    print("ref=", ref)
    print("t=", t)
    return t

@testcase(name="zeros: shape=[3]", score=10)

def zeros1(impl):
    t = impl.zeros([3])
    return t

@testcase(name="zeros_like: shape=[2,1]", score=10)

def zeros_like1(impl):
    ref = impl.ones([2, 1])
    t = impl.zeros_like(ref)
    return t

@testcase(name="randn: shape=[4]", score=10)

def randn1(impl):
    t = impl.randn([4])
    return t.shape

@testcase(name="randn_like: shape=[2,2]", score=10)

def randn_like1(impl):
    ref = impl.ones([2, 2])
    t = impl.randn_like(ref)
    ## Can't check the values of the tensor, but we can check the shape
    return t.shape

@testcase(name="empty: shape=[2,3]", score=10)

def empty1(impl):
    t = impl.empty([2, 3])
    return t.shape

@testcase(name="empty_like: shape=[1]", score=10)

def empty_like1(impl):
    ref = impl.ones([1])
    t = impl.empty_like(ref)
    return t.shape

@testcase(name="arange: 2, 6", score=10)

def arange1(impl):
    t = impl.arange(2, 6)
    return t

@testcase(name="arange: 2, 10, 3", score=10)

def arange2(impl):
    t = impl.arange(2, 10, 3)
    return t

@testcase(name="range: 2, 6", score=10)

def range1(impl):
    t = impl.range(2, 6)
    return t

@testcase(name="range: 2, 10, 3", score=10)

def range2(impl):
    t = impl.range(2, 10, 3)
    return t

@testcase(name="linspace: 1, 3, 3", score=10)

def linspace1(impl):
    t = impl.linspace(1, 3, 3)
    return t

@testcase(name="linspace: -1, 1, 5", score=10)

def linspace2(impl):
    t = impl.linspace(-1, 1, 5)
    return t


def testsets_part8():
    ones1()
    ones_like1()
    ones_like_scalar()
    zeros1()
    zeros_like1()
    randn1()
    randn_like1()
    empty1()
    empty_like1()
    arange1()
    arange2()
    range1()
    range2()
    linspace1()
    linspace2()
    
if __name__ == "__main__":
    print("Beginning grading part8")
    set_debug_mode(True)
    testsets_part8()
    grader_summary("part8")