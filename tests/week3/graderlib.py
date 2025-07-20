import sys
import os
from functools import wraps
import numpy as np
import traceback

self_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(self_path + "/../../")
sys.path.append(self_path)

import clownpiece as CP
from clownpiece import Tensor
from clownpiece.nn import *
from clownpiece.nn.module import Parameter, Buffer
from clownpiece.autograd import no_grad

from typing import Tuple, Any, Iterable
import multiprocessing as mp

# global vars
total_score: int = 0
passed_score: int = 0
failed_test = []
passed_test = []

DEBUG_MODE = False

# utils

# EXPORT
"""
  Set debug mode to True to raise exceptions directly (to see full backtrace)
"""
def set_debug_mode(debug: bool):
  global DEBUG_MODE
  DEBUG_MODE = debug
  
"""
  Use DEBUG env var to enable DEBUG mode
"""
if os.getenv("DEBUG", None) is not None:
  set_debug_mode(True)

def exec_with_timeout(func, *args, timeout=None, **kwargs) -> Tuple[bool, Any]:
    try:
        result = func(*args, **kwargs)
        if DEBUG_MODE:
            print("result:", result)
        return True, result
    except Exception as e:
        if DEBUG_MODE:
            print("Exception in execution:", e)
            traceback.print_exc()  # 打印完整Traceback
            raise e
        return False, str(e)

def print_separate_line():
  print("=" * 50)

def tensor_close(t1: Tensor, t2: Tensor, rtol=1e-4, atol=1e-6) -> bool:
    """Check if two tensors are close within tolerance"""
    if not isinstance(t1, Tensor) or not isinstance(t2, Tensor):
        return False
    
    if tuple(t1.shape) != tuple(t2.shape):
        return False
    
    # Convert to numpy for comparison
    arr1 = np.array(t1.tolist())
    arr2 = np.array(t2.tolist())
    diff = np.abs(arr1 - arr2)
    return np.all(diff <= atol + rtol * np.abs(arr2))

def value_close(v1, v2, rtol=1e-4, atol=1e-6) -> bool:
    """Check if two values are close, handling different types"""
    if type(v1) != type(v2):
        return False
    
    if isinstance(v1, Tensor):
        return tensor_close(v1, v2, rtol, atol)
    elif isinstance(v1, (list, tuple)):
        if len(v1) != len(v2):
            return False
        return all(value_close(a, b, rtol, atol) for a, b in zip(v1, v2))
    elif isinstance(v1, dict):
        if set(v1.keys()) != set(v2.keys()):
            return False
        return all(value_close(v1[k], v2[k], rtol, atol) for k in v1.keys())
    elif isinstance(v1, (int, float)):
        return abs(v1 - v2) <= atol + rtol * abs(v2)
    else:
        return v1 == v2

def testcase(name: str, score: int, timeout: int = 30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global total_score, passed_score, failed_test, passed_test
            
            total_score += score
            print(f"Running {name}...")
            
            success, result = exec_with_timeout(func, timeout=timeout, *args, **kwargs)
            
            if success:
                passed_score += score
                passed_test.append(name)
                print(f"✓ {name} passed ({score} points)")
                return result
            else:
                failed_test.append((name, result))
                print(f"✗ {name} failed: {result}")
                return None
                
        return wrapper
    return decorator

def grader_summary():
    print_separate_line()
    print(f"Total Score: {passed_score}/{total_score}")
    print(f"Passed: {len(passed_test)}")
    print(f"Failed: {len(failed_test)}")
    
    if failed_test:
        print("\nFailed tests:")
        for test_name, error in failed_test:
            print(f"  - {test_name}: {error}")
    
    if passed_test:
        print(f"\nPassed tests: {', '.join(passed_test)}")
    
    print_separate_line()
    return passed_score, total_score
