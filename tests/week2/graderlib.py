import sys
import os


self_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(self_path + "/../../")
sys.path.append(self_path)

import clownpiece as CT
from clownpiece import Tensor as TensorCT
import torch as T
from torch import Tensor as TensorT

from typing import Tuple, Any, Iterable
import torch.multiprocessing as mp
from clownpiece.utils_ import wrap_tuple

# global vars
total_score: int = 0
passed_score: int = 0
failed_test = []
passed_test = []

DEBUG_MODE = False

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

def exec_with_timeout_mp(func, *args, timeout=None, **kwargs) -> Tuple[bool, Any]:

  with mp.Manager() as manager:
    # why manager? pytorch mp treat pickling tensor in special way, if the worker quit, it's no longer available
    def worker(result_queue, func):
      try:
        # execute the target funcction
        result = func(*args, **kwargs)
        
        results = wrap_tuple(result)
        # detach torch tensor so that non-leaf tensor can be pickled.
        results = [r.detach() if isinstance(r, TensorT) else r for r in results]
        
        result_queue.put((True, results))
      except Exception as e:
        if DEBUG_MODE:
          result_queue.put((False, str(e)))
          raise e
        else:
          result_queue.put((False, str(e)))

    result_queue = manager.Queue()
    process = mp.Process(target=worker, args=(result_queue, func))
    process.start()
    process.join(timeout)

    if process.is_alive():
      process.terminate()
      return False, RuntimeError(f"Timeout after {timeout} seconds; terminated")

    success, result = result_queue.get()
  return success, result

def exec_with_timeout(func, *args, timeout=None, **kwargs) -> Tuple[bool, Any]:
    try:
        result = func(*args, **kwargs)
        if DEBUG_MODE:
            print("result:", result)
        return True, result
    except Exception as e:
        if DEBUG_MODE:
            print("Exception in execution:", e)
            raise e
        return False, str(e)

def print_separate_line():
  print("=" * 50)

def all_close(t1: TensorCT, t2: TensorT)->bool:
  is_tensor_t1 = isinstance(t1, (TensorCT, TensorT))
  is_tensor_t2 = isinstance(t2, (TensorCT, TensorT))
  
  if is_tensor_t1 != is_tensor_t2:
    print(f"all_close: type mismatch: {type(t1)} vs {type(t2)}")
    return False
  
  if not is_tensor_t1 or not is_tensor_t2:
    return t1 == t2
  
  eps = 1e-4
  
  if tuple(t1.shape) != tuple(t2.shape):
    print(f"all_close: shape mismatch: {t1.shape} vs {t2.shape}")
    return False
  
  t1 = t1.reshape(-1,)
  t2 = t2.reshape(-1,)
  for i in range(len(t1)):
    if abs(t1[i].item() - t2[i].item()) > eps:
      print(f"all_close: value mismatch at index {i}: {t1[i].item()} vs {t2[i].item()}")
      return False
  
  return True

def check_result_match(t1_list: Iterable[TensorCT], t2_list: Iterable[TensorT])->bool:
  t1_list = wrap_tuple(t1_list)
  t2_list = wrap_tuple(t2_list)
  if len(t1_list) != len(t2_list):
    return False
  for t1, t2 in zip(t1_list, t2_list):
    if not all_close(t1, t2):
      return False
  return True

# EXPORT
"""
  Decorator around a Tensor computation function. Return result will be compared against pytorch.
"""
def testcase(name: str, score: int, timeout:int = 60):

  
  def decorator(func): # expect function to take only one args: impl (either clownpiece or torch module)
    def wrapper():
      global total_score
      total_score += score
      
      print_separate_line()
      print(f"{name}: test start")
      
      global passed_score, failed_test, passed_test
      success_ct, result_ct = exec_with_timeout(func,impl=CT, timeout=timeout)
      success_t, result_t = exec_with_timeout(func,impl=T, timeout=timeout)

      if not success_ct:
        print(f"{name}: test failed | test program failed with error:\n\n"
              f"{result_ct}\n\n"
              f"> Set DEBUG mode to see full traceback\n"
              )
        failed_test += [name]
        return
        
      if not success_t:
        print(f"{name}: test failed | reference program failed with error:\n\n"
              f"{result_t}\n\n"
              f"> Set DEBUG mode to see full traceback\n"
              f"> This should not happen; please contact TA with traceback log !!!")
        failed_test += [name]
        return
      
      if not check_result_match(result_ct, result_t):
        print(f"{name}: test failed | test output does not match reference output")
        print(f">test outputs:\n{result_ct}\n")
        print(f">reference outputs:\n{result_t}\n")
        failed_test += [name]
        return
      
      print(f"{name}: test passed")
      
      passed_score += score
      passed_test += [name]
  
    return wrapper
  return decorator
    
def grader_summary(name):
  print_separate_line()
  print(f"{name}: grader summary:")
  print(f"score: {passed_score}/{total_score}")
  print(f"# passed: {len(passed_test)}")
  print(f"# failed: {len(failed_test)}")
  if failed_test:
    print("failed tests:", ", ".join(failed_test))
  else:
    print("all tests passed!")
  print_separate_line()