# wrap x into tuple if it's not already
def wrap_tuple(x):
  return (x,) if not isinstance(x, (list, tuple)) else tuple(x)

# Returns the ceiling of the division of num by den.
def ceil_div(num: int, den: int) -> int:
  return (num + den - 1) // den if den != 0 else 0