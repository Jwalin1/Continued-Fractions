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


# Functions to convert between binary and decimal.
def frac_dec_to_bin(x: float, preferred_digits=200):
    assert 0 < x < 1
    binary_digits = []
    for _ in range(preferred_digits):
        x *= 2
        bin_digit = int(x)
        binary_digits.append(bin_digit)
        x -= bin_digit
    return binary_digits

def frac_bin_to_dec(digits: list[int]):
    x = mp.mpf()
    pow_2 = mp.mpf('1')
    for digit in digits:
        pow_2 *= 0.5
        if digit == 1:
            x += pow_2
        elif digit == 0:
            pass
        else:
            raise ValueError('Not a binary digit.')
    return x

def compute_binary_diff_coeffs(x, preferred_digits=200):
    assert 0 < x < 1
    bin_cf = np.array(frac_dec_to_bin(x, preferred_digits))
    bin_cf = np.insert(bin_cf, 0, 1)
    return np.diff(np.where(bin_cf)[0])

def diff_to_bin(coeffs):
    binary_repr = []
    for coeff in coeffs:
        binary_repr.extend([0]*(coeff-1))
        binary_repr.append(1)
    return binary_repr
