import numpy as np
import mpmath as mp


def compute_khinchin_const(p: float, n: int):
  k = np.arange(1, n, dtype=float)
  sum_ = (-(k**p) * np.log2(1-1/((k+1)**2))).sum()
  return sum_ ** (1/p)


def Brjuno_func(convergents: list[tuple[int, int]]) -> float:
  sum_ = 0
  for i in range(len(convergents)-1):
    sum_ += mp.log(convergents[i+1][-1]) / convergents[i][-1]
  return sum_

def Yoccoz_func(fracs: np.ndarray[int]) -> float:
  sum_ = 0
  for i in range(len(fracs)):
    sum_ += np.prod(fracs[:i]) * mp.log(1/fracs[i])
  return sum_
