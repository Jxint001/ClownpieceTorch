import sys
import os


import torch
from graderlib import self_path
from graderlib import set_debug_mode, testcase, grader_summary



@testcase(name="permute1: simple permutation", score=10)

def permute1(impl=torch):
    # Ask fAKe for detailed information about "some dimensions may be missing"
    a = impl.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    
    return (impl.permute(a, (0, 2, 1)), 
            impl.permute(a, (1, 0, 2)),
            impl.permute(a, (1, 2, 0)),
            impl.permute(a, (2, 0, 1)))

@testcase(name="transpose1: simple transpose", score=10)

def transpose1(impl=torch):
    a = impl.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    
    return (impl.transpose(a, dim0 = 0, dim1 = 1), 
            impl.transpose(a, dim0 = 1, dim1 = 1),
            impl.transpose(a, dim0 = 0, dim1 = 2),
            impl.transpose(a, dim0 = 1, dim1 = 2))
    
@testcase(name="transpose2: is_contiguous() and clone after transpose", score=10)

def transpose2(impl=torch):
    a = impl.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    
    b1 = impl.transpose(a, dim0 = 0, dim1 = 1)
    b2 = impl.transpose(a, dim0 = 1, dim1 = 2)
    b3 = impl.transpose(a, dim0 = 0, dim1 = 2)
    
    c1 = b1.contiguous()
    c2 = b2.contiguous()
    c3 = b3.contiguous()
    
    return (b1, b2, b3, b1.is_contiguous(), b2.is_contiguous(), b3.is_contiguous(),
            c1, c2, c3, c1.is_contiguous(), c2.is_contiguous(), c3.is_contiguous())
    
    # d1 = b1.clone()   # disabled due to PyTorch's clone behavior
    # d2 = b2.clone()
    # d3 = b3.clone()
    
    # return (b1, b2, b3, b1.is_contiguous(), b2.is_contiguous(), b3.is_contiguous(),
    #         c1, c2, c3, c1.is_contiguous(), c2.is_contiguous(), c3.is_contiguous(),
    #         d1, d2, d3, d1.is_contiguous(), d2.is_contiguous(), d3.is_contiguous())

@testcase(name="reshape1_noerror: all reshape/view should succeed", score=10)

def reshape1_noerror(impl=torch):
    # print("reshape1=============================================", flush=True)
    
    a0 = impl.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    r0 = a0.reshape(-1,)
    
    # contiguous tensor, view possible, no -1
    a = impl.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    r1 = a.reshape((4, 2))
    v1 = a.view((4, 2))
    
    # contiguous tensor, view possible, with -1
    r2 = a.reshape((2, -1))
    v2 = a.view((2, -1))
    
    # reshape to flat, still view for contiguous
    r3 = a.reshape((8,))
    
    # non-contiguous source, reshape will copy
    b = impl.transpose(a, dim0 = 0, dim1 = 1)
    r4 = b.reshape((4, 2))

    # -1 deduction, covering only one -1
    c = impl.Tensor([1, 2, 3, 4, 5, 6])
    r6 = c.reshape((-1, 2))
    v4 = c.view((-1, 2))
    # print("reshape1=============================================", flush=True)
    return (r0, r1, v1, r2, v2, r3, r4, r6, v4)

@testcase(name="reshape1_error: reshape/view with expected error", score=10)

def reshape1_error(impl=torch):
    #print("reshapeee=============================================", flush=True)
    # view on non-contiguous
    a = impl.Tensor([[1,2],[3,4],[5,6]])
    d2 = impl.transpose(a, dim0 = 0, dim1 = 1)
    try:
        v5 = d2.view((6,))
        return False
    except RuntimeError:
        pass
    
    # -1 deduction: more than one -1
    c = impl.Tensor([1, 2, 3, 4, 5, 6])
    try:
        r7 = c.reshape((-1, -1))
        return False
    except RuntimeError:
        pass

    # shape does match numel
    try:
        r8 = c.reshape((5, 2))
        return False
    except RuntimeError:
        return True

@testcase(name="reshape2: reshape/view edge cases", score=10)

def reshape2(impl=torch):
    #print("reshape2=============================================", flush=True)
    # Reshape of scalar to (1,) and back (0-dimensional tensor)
    if impl.__name__ == "torch":
        a = impl.tensor(3)
    else:
        a = impl.Tensor(3)
    #print("start reshape", flush=True)
    r1 = a.reshape(())
    #print("ok", flush=True)
    v1 = a.view(())
    r2 = a.reshape((1,))
    v2 = a.view((1,))
    
    # Reshape 1D to 2D with singleton
    b = impl.Tensor([1])
    r3 = b.reshape((1, 1),)
    v3 = b.view((1, 1))

    return (r1, v1, r2, v2, r3, v3)

@testcase(name="narrow1_noerror: narrow views succeed", score=10)

def narrow1_noerror(impl=torch):
    a = impl.Tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])

    r1 = impl.narrow(a, 0, 0, 2)
    r2 = impl.narrow(a, 1, 2, 3)
    r3 = impl.narrow(a, 0, 3, 1)
    r4 = impl.narrow(a, 1, 0, 5)
    
    # higher dimensional tensor
    b = impl.Tensor([[[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11]],
                     [[12, 13, 14, 15],
                      [16, 17, 18, 19],
                      [20, 21, 22, 23]]])
    r5 = impl.narrow(b, 1, 1, 2)
    # 0 length
    r6 = impl.narrow(a, 1, 3, 0)
    return (r1, r2, r3, r4, r5, r6)

@testcase(name="narrow1_error: narrow with expected error", score=10)

def narrow1_error(impl=torch):
    a = impl.Tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    # dim out of range
    try:
        impl.narrow(a, 2, 0, 1)
        return False
    except Exception:
        pass
    # start out of range
    try:
        impl.narrow(a, 0, 5, 1)
        return False
    except Exception:
        pass
    # length out of range
    try:
        impl.narrow(a, 1, 5, 2)
        return False
    except Exception:
        return True

@testcase(name="chunk1: chunk on dim 0, divisible", score=10)

def chunk1(impl=torch):
    a = impl.Tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    return impl.chunk(a, 2, 0)  # 应返回2个(2,6)的tensor

@testcase(name="chunk2: chunk on dim 1, divisible", score=10)

def chunk2(impl=torch):
    a = impl.Tensor([[ 0,  1,  2,  3,  4,  5],
                     [ 6,  7,  8,  9, 10, 11],
                     [12, 13, 14, 15, 16, 17],
                     [18, 19, 20, 21, 22, 23]])
    return impl.chunk(a, 3, 1)  # 应返回3个(4,2)的tensor

@testcase(name="chunk3: chunk on dim 1, not divisible", score=10)

def chunk3(impl=torch):
    b = impl.Tensor([[ 0,  1,  2,  3,  4,  5,  6],
                     [ 7,  8,  9, 10, 11, 12, 13]])
    return impl.chunk(b, 3, 1)  # 应返回3个(2, 3/2/2)的tensor

@testcase(name="chunk4: chunk on dim 0, not divisible", score=10)

def chunk4(impl=torch):
    b = impl.Tensor([[ 0,  1,  2,  3,  4,  5,  6],
                     [ 7,  8,  9, 10, 11, 12, 13]])
    return impl.chunk(b, 4, 0)  # 应返回4个，每个shape (0或1,7)

@testcase(name="chunk5: chunk 1d tensor", score=10)

def chunk5(impl=torch):
    d = impl.Tensor([0, 1, 2, 3, 4])
    return impl.chunk(d, 2, 0)  # 应返回2个(3,)和(2,)的tensor

@testcase(name="chunk1_error: chunk with expected error", score=10)

def chunk_error(impl=torch):
    a = impl.Tensor([[ 0,  1,  2,  3],
                     [ 4,  5,  6,  7],
                     [ 8,  9, 10, 11]])
    # dim out of range
    try:
        impl.chunk(a, 2, 5)
        return False
    except Exception:
        pass
    # chunks <= 0
    try:
        impl.chunk(a, 0, 0)
        return False
    except Exception:
        pass
    try:
        impl.chunk(a, -1, 1)
        return False
    except Exception:
        return True

@testcase(name="split1", score=10)

def split1(impl=torch):
    # split int divisible
    x = impl.Tensor([[i + j * 4 for i in range(4)] for j in range(6)])  # shape (6,4)
    return impl.split(x, 2, 0)  # split_size=2, dim=0，=> 3 chunks (2,4)

@testcase(name="split2", score=10)

def split2(impl=torch):
    # split int not divisible
    x = impl.Tensor([[i + j * 3 for i in range(3)] for j in range(5)])  # shape (5,3)
    return impl.split(x, 2, 0)  # split_size=2, dim=0，=> 3 chunks [(2,3),(2,3),(1,3)]

@testcase(name="split3", score=10)

def split3(impl=torch):
    # split int dim 1
    x = impl.Tensor([[i + j * 7 for i in range(7)] for j in range(2)])  # shape (2,7)
    return impl.split(x, 3, 1)  # split_size=3, dim=1，=> 3 chunks [(2,3),(2,3),(2,1)]

@testcase(name="split4", score=10)

def split4(impl=torch):
    # split int size larger than dim
    x = impl.Tensor([[1, 2, 3]])
    return impl.split(x, 10, 1)  # split_size=10 > shape[1]，=> 1 chunk [(1,3)]

@testcase(name="split5", score=10)

def split5(impl=torch):
    # split int size is 1
    x = impl.Tensor([0, 1, 2, 3])
    return impl.split(x, 1, 0)  # split_size=1, dim=0，=> 4 chunks [(1,), (1,), (1,), (1,)]

@testcase(name="split6", score=10)

def split6(impl=torch):
    # split sections basic
    x = impl.Tensor([[i + j * 4 for i in range(4)] for j in range(6)])  # shape (6,4)
    return impl.split(x, [2, 3, 1], 0)  # sections=[2,3,1], dim=0

@testcase(name="split7", score=10)

def split7(impl=torch):
    # split sections dim 1
    x = impl.Tensor([[i + j * 7 for i in range(7)] for j in range(2)])  # shape (2,7)
    return impl.split(x, [2, 2, 3], 1)  # sections=[2,2,3], dim=1

@testcase(name="split8", score=10)

def split8(impl=torch):
    # split sections single section
    x = impl.Tensor([1, 2, 3, 4])
    return impl.split(x, [4], 0)  # sections=[4], dim=0

@testcase(name="split9", score=10)

def split9(impl=torch):
    # split sections dim size not match
    x = impl.Tensor([0, 1, 2])
    try:
        return impl.split(x, [1, 1], 0)  # 总和不等于3，应抛异常
    except Exception:
        return True

@testcase(name="stack1", score=10)

def stack1(impl=torch):
    # stack along new dimension dim=0
    a = impl.Tensor([1, 2, 3])
    b = impl.Tensor([4, 5, 6])
    case1 = impl.stack([a, b], 0)  # shape (2,3)

    # stack along new dimension dim=1
    case2 = impl.stack([a, b], 1)  # shape (3,2)

    # stack with negative dim
    case3 = impl.stack([a, b], -1)  # shape (3,2)

    # stack higher dimensional tensor along axis 2
    a2 = impl.Tensor([[1, 2], [3, 4]])
    b2 = impl.Tensor([[5, 6], [7, 8]])
    case4 = impl.stack([a2, b2], 2)  # shape (2,2,2)

    # stack single tensor
    case5 = impl.stack([a], 0)  # shape (1,3)

    # stack with shape mismatch, should raise
    a3 = impl.Tensor([1, 2, 3])
    b3 = impl.Tensor([4, 5])
    try:
        case6 = impl.stack([a3, b3], 0)
    except Exception:
        case6 = "raises"
    else:
        case6 = "not raise"

    return case1, case2, case3, case4, case5, case6

@testcase(name="cat1", score=10)

def cat1(impl=torch):
    # cat along existing dim=0
    a = impl.Tensor([[1, 2]])
    b = impl.Tensor([[3, 4]])
    case1 = impl.cat([a, b], 0)  # shape (2,2)

    # cat along existing dim=1
    a2 = impl.Tensor([[1], [2]])
    b2 = impl.Tensor([[3], [4]])
    case2 = impl.cat([a2, b2], 1)  # shape (2,2)

    # cat along negative dim
    case3 = impl.cat([a, b], -2)  # shape (2,2)

    # cat 3D tensors along dim=1
    a3 = impl.Tensor([[[1, 2]], [[3, 4]]])
    b3 = impl.Tensor([[[5, 6]], [[7, 8]]])
    case4 = impl.cat([a3, b3], 1)  # shape (2,2,2)

    # cat single tensor
    a4 = impl.Tensor([[1, 2, 3]])
    case5 = impl.cat([a4], 0)  # shape (1,3)

    # cat with shape mismatch, should raise
    a5 = impl.Tensor([[1, 2, 3]])
    b5 = impl.Tensor([[4, 5]])
    try:
        case6 = impl.cat([a5, b5], 1)
    except Exception:
        case6 = "raises"
    else:
        case6 = "not raise"

    return case1, case2, case3, case4, case5, case6

@testcase(name="squeeze_unsqueeze_all_cases", score=10)

def squeeze_unsqueeze_all_cases(impl):
    results = []

    # --- SQUEEZE CASES ---

    # case1: Squeeze at dim=0, shape=(1,3,4) → (3,4)
    a = impl.Tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])  # (1,3,4)
    case1 = impl.squeeze(a, 0)
    results.append(case1)

    # case2: Squeeze at dim=1, shape=(2,1,4) → (2,4)
    b = impl.Tensor([[[1,2,3,4]], [[5,6,7,8]]])  # (2,1,4)
    case2 = impl.squeeze(b, 1)
    results.append(case2)

    # case3: Squeeze at negative dim, shape=(2,4,1), dim=-1 → (2,4)
    c = impl.Tensor([[[1],[2],[3],[4]], [[5],[6],[7],[8]]])  # (2,4,1)
    case3 = impl.squeeze(c, -1)
    results.append(case3)

    # case4: Squeeze with dim not 1, should raise
    # This case is disabled due to differences in PyTorch versions.
    # Trying to squeeze a dimension that is not 1 would not raise an error in PyTorch.
    # d = impl.Tensor([[1,2,3],[4,5,6]])  # (2,3)
    # try:
    #     case4 = impl.squeeze(d, 1)
    # except Exception:
    #     case4 = "raises"
    # results.append(case4)

    # case5: Squeeze only dimension, shape=(1,), dim=0
    e = impl.Tensor([42])  # (1,)
    case5 = impl.squeeze(e, 0)
    results.append(case5)

    # case7: Squeeze with negative dim out of range
    try:
        case7 = impl.squeeze(a, -4)
    except Exception:
        case7 = "raises"
    results.append(case7)

    # --- UNSQUEEZE CASES ---

    # case8: Unsqueeze at dim=0, shape=(3,4) → (1,3,4)
    g = impl.Tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])  # (3,4)
    case8 = impl.unsqueeze(g, 0)
    results.append(case8)

    # case9: Unsqueeze at dim=1, shape=(3,4) → (3,1,4)
    case9 = impl.unsqueeze(g, 1)
    results.append(case9)

    # case10: Unsqueeze at dim=2, shape=(3,4) → (3,4,1)
    case10 = impl.unsqueeze(g, 2)
    results.append(case10)

    # case11: Unsqueeze at negative dim, -1 for (3,4) should insert before last, → (3,4,1)
    case11 = impl.unsqueeze(g, -1)
    results.append(case11)

    # case12: Unsqueeze at dim=len+1 (invalid), should raise
    try:
        case12 = impl.unsqueeze(g, 3)
    except Exception:
        case12 = "raises"
    results.append(case12)

    # case13: Unsqueeze at dim=-3 (invalid), should raise
    try:
        case13 = impl.unsqueeze(g, -3)
    except Exception:
        case13 = "raises"
    results.append(case13)

    # case14: Unsqueeze scalar (no dimension) at 0
    if impl.__name__ == "torch":
        h = impl.tensor(5)
    else:
        h = impl.Tensor(5)  # scalar
    case14 = impl.unsqueeze(h, 0)
    results.append(case14)

    return results

@testcase(name="broadcast_to_all_cases", score=10)

def broadcast_to_all_cases(impl):
    results = []

    # case1: Scalar broadcast to (2,3)
    if impl.__name__ == "torch":
        a = impl.tensor(5)
    else:
        a = impl.Tensor(5)  # shape=()
    try:
        case1 = impl.broadcast_to(a, (2, 3))  # shape (2,3)
    except Exception:
        case1 = "raises"
    results.append(case1)

    # case2: (1,3) broadcast to (4,3)
    b = impl.Tensor([[1,2,3]])  # shape=(1,3)
    try:
        case2 = impl.broadcast_to(b, (4,3))  # (4,3)
    except Exception:
        case2 = "raises"
    results.append(case2)

    # case3: (2,1,1) broadcast to (2,4,3)
    c = impl.Tensor([[[7]], [[8]]])  # (2,1,1)
    try:
        case3 = impl.broadcast_to(c, (2,4,3))  # (2,4,3)
    except Exception:
        case3 = "raises"
    results.append(case3)

    # case4: (3,) broadcast to (2,3)
    d = impl.Tensor([1,2,3])  # (3,)
    try:
        case4 = impl.broadcast_to(d, (2,3))  # (2,3)
    except Exception:
        case4 = "raises"
    results.append(case4)

    # case5: (2,3) broadcast to (2,3) (no change)
    e = impl.Tensor([[1,2,3],[4,5,6]])
    try:
        case5 = impl.broadcast_to(e, (2,3))  # (2,3)
    except Exception:
        case5 = "raises"
    results.append(case5)

    # case6: (2,3) broadcast to (3,2,3)
    try:
        case6 = impl.broadcast_to(e, (3,2,3))
    except Exception:
        case6 = "raises"
    results.append(case6)

    # case7: (2,3) broadcast to (2,3,4) is invalid, should raise
    try:
        case7 = impl.broadcast_to(e, (2,3,4))
    except Exception:
        case7 = "raises"
    results.append(case7)

    # case8: (1,3,1) broadcast to (2,3,4) is invalid, should raise
    f = impl.Tensor([[[1],[2],[3]]])  # (1,3,1)
    try:
        case8 = impl.broadcast_to(f, (2,3,4))
    except Exception:
        case8 = "raises"
    results.append(case8)

    # case9: (3,) broadcast to (3,) (no change)
    g = impl.Tensor([1,2,3])
    try:
        case9 = impl.broadcast_to(g, (3,))
    except Exception:
        case9 = "raises"
    results.append(case9)

    return results

@testcase(name="broadcast_pair_case1", score=10)

def broadcast_pair_case1(impl):
    a = impl.Tensor([[1], [2]])     # shape=(2,1)
    b = impl.Tensor([[10, 20, 30]]) # shape=(1,3)
    try:
        result = impl.broadcast_tensors(a, b)
    except Exception:
        result = "raises1"
    return result

@testcase(name="broadcast_pair_case2", score=10)

def broadcast_pair_case2(impl):
    c = impl.Tensor([100, 200, 300])   # shape=(3,)
    d = impl.Tensor([[4, 5, 6]])       # shape=(1,3)
    try:
        result = impl.broadcast_tensors(c, d)
    except Exception:
        result = "raises2"
    return result

@testcase(name="broadcast_pair_case3", score=10)

def broadcast_pair_case3(impl):
    e = impl.Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
    f = impl.Tensor([[7], [8]])              # (2,1)
    try:
        result = impl.broadcast_tensors(e, f)
    except Exception:
        result = "raises3"
    return result

@testcase(name="broadcast_pair_case4", score=10)

def broadcast_pair_case4(impl):
    e = impl.Tensor([[1, 2, 3], [4, 5, 6]]) # shape=(2,3)
    c = impl.Tensor([100, 200, 300])        # shape=(3,)
    try:
        result = impl.broadcast_tensors(e, c)
    except Exception:
        result = "raises4"
    return result

@testcase(name="broadcast_pair_case5", score=10)

def broadcast_pair_case5(impl):
    e = impl.Tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
    g = impl.Tensor([[1,2,3,4],[5,6,7,8]])   # (2,4)
    try:
        result = impl.broadcast_tensors(e, g)
    except Exception:
        result = "raises5"
    return result

@testcase(name="broadcast_pair_case6", score=10)

def broadcast_pair_case6(impl):
    if impl.__name__ == "torch":
        r = impl.tensor(42)
        s = impl.tensor(24)
    else:
        r = impl.Tensor(42)
        s = impl.Tensor(24)
    try:
        result = impl.broadcast_tensors(r, s)
    except Exception:
        result = "raises6"
    return result

def testsets_part7():
    permute1()
    sys.stdout.flush()

    transpose1()
    sys.stdout.flush()
    transpose2() 
    sys.stdout.flush()
    # Part of testcase transpose2() is disabled due to differences in PyTorch versions.
    # Note: PyTorch's clone automatically makes the tensor contiguous, so the test is not valid.
    
    reshape1_noerror()
    sys.stdout.flush()
    reshape1_error()
    sys.stdout.flush()
    reshape2()
    sys.stdout.flush()
    
    narrow1_noerror()
    sys.stdout.flush()
    narrow1_error()
    sys.stdout.flush()
    
    chunk1()
    sys.stdout.flush()
    chunk2()
    sys.stdout.flush()
    chunk3()
    sys.stdout.flush()
    chunk4()
    sys.stdout.flush()
    chunk5()
    sys.stdout.flush()
    chunk_error()
    sys.stdout.flush()
    
    split1()
    sys.stdout.flush()
    split2()
    sys.stdout.flush()
    split3()
    sys.stdout.flush()
    split4()
    sys.stdout.flush()
    split5()
    sys.stdout.flush()
    split6()
    sys.stdout.flush()
    split7()
    sys.stdout.flush()
    split8()
    sys.stdout.flush()
    split9()
    sys.stdout.flush()
    
    stack1()
    sys.stdout.flush()
    cat1()
    sys.stdout.flush()
    
    squeeze_unsqueeze_all_cases()
    sys.stdout.flush()
    broadcast_to_all_cases()
    sys.stdout.flush()
    
    broadcast_pair_case1()
    sys.stdout.flush()
    broadcast_pair_case2()
    sys.stdout.flush()
    broadcast_pair_case3()
    sys.stdout.flush()
    broadcast_pair_case4()
    sys.stdout.flush()
    broadcast_pair_case5()
    sys.stdout.flush()
    broadcast_pair_case6()
    sys.stdout.flush()
    
    
if __name__ == "__main__":
    print("Beginning grading part7")
    # set_debug_mode(True)
    testsets_part7()
    grader_summary("part7")