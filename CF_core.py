import numpy as np
import mpmath as mp
from scipy import stats


def compute_num_from_cf(coeffs: list[int]) -> float:
  num = mp.mpf(float(coeffs[-1]))
  for coeff in coeffs[-2::-1]:
    num = coeff + 1/num
  return num

def compute_cf(num: float, num_coeffs: int) -> np.ndarray[int]:
  coeffs = []
  for _ in range(num_coeffs):
    a = mp.floor(num)
    try:
      coeffs.append(np.int64(a))  # Num too big.
      num = 1 / (num - a)  # Zero div.
    except (ZeroDivisionError, OverflowError):
      break
  return np.array(coeffs)


def _update_convergent(
    coeff: int, h_prev: int, h_curr: int, k_prev: int, k_curr: int
    ) -> tuple[int, int, int, int, int, int]:
    h_next = coeff*h_curr + h_prev
    k_next = coeff*k_curr + k_prev
    h_prev, h_curr = h_curr, h_next
    k_prev, k_curr = k_curr, k_next
    return h_next, h_curr, h_prev, k_next, k_curr, k_prev

def compute_convergents_from_cf(
    coeffs: list[int], all_convergents: bool = True
) -> tuple[int, int] | list[tuple[int, int]]:
    h_prev, h_curr = mp.mpf(0), mp.mpf(1)
    k_prev, k_curr = mp.mpf(1), mp.mpf(0)
    convergents = []
    for coeff in coeffs:
        h_next, h_curr, h_prev, k_next, k_curr, k_prev = _update_convergent(coeff, h_prev, h_curr, k_prev, k_curr)
        if all_convergents:
            convergents.append((h_next, k_next))
    return convergents if all_convergents else (h_next, k_next)


# Functions for the Khinchin's constant.
# compute_cum_GM = lambda x: [stats.gmean(x[:i]) for i in range(1,1+len(x))]
compute_cum_GM = lambda x: np.exp(np.cumsum(np.log(x)) / np.arange(1,1+len(x)))
compute_cum_HM = lambda x: np.arange(1, 1+len(x)) / np.cumsum(1/x)
def compute_CF_GMs(num: float, num_coeffs: int) -> np.ndarray[float]:
  return compute_cum_GM(compute_cf(num, 1+num_coeffs)[1:])

# Functions for the Lévy constant.
compute_successive_roots = lambda convergents: np.array(
  [convergent[-1]**(1/i) for i,convergent in enumerate(convergents, start=1)], dtype=float)
def compute_convergent_denom_roots(num: float, num_coeffs: int) -> np.ndarray[float]:
  convergents = compute_convergents_from_cf(
    compute_cf(num, num_coeffs+1)[1:], all_convergents=True)
  return compute_successive_roots(convergents)

# Function for Yoccoz function.
def compute_successive_frac_invs(num: float, n: int):
  coeffs = [num%1]
  for _ in range(n-1):
    if coeffs[-1] == 0:
      break
    coeffs.append((1/coeffs[-1])%1)
  return coeffs
