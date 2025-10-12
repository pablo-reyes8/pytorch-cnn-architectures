# tests/test_utils_scaling.py
import math
import torch
import pytest
from model.cnn_utils import *


def test_make_divisible_basic():
    assert make_divisible(15, 8) == 16
    assert make_divisible(8, 8) == 8
    assert make_divisible(1, 8) == 8
    # no caer >10% por debajo
    v = 33
    out = make_divisible(v, 8)
    assert out >= 0.9 * v

@pytest.mark.parametrize("width_mult,c_out,exp", [
    (1.0, 40, 40),
    (1.1, 40, 48),   # 44 -> redondeo a 48 (múltiplo de 8)
    (0.9, 40, 40),   # 36 -> bump para no caer >10%
])
def test_round_filters(width_mult, c_out, exp):
    assert round_filters(c_out, width_mult, divisor=8) == exp

@pytest.mark.parametrize("depth_mult,n,exp", [
    (1.0, 2, 2),
    (1.2, 2, 3),
    (1.49, 3, 5),
])
def test_round_repeats(depth_mult, n, exp):
    assert round_repeats(n, depth_mult) == exp

def test_round_resolution_divisible_and_monotonic():
    base = 200
    a = round_resolution(base, 1.00, 8)
    b = round_resolution(base, 1.15, 8)
    c = round_resolution(base, 1.15**2, 8)
    assert a % 8 == 0 and b % 8 == 0 and c % 8 == 0
    assert a <= b <= c
