import numpy as np
import pytest

from xeofs.inumpy import EOF

rng = np.random.default_rng(7)

m = 10
n = 5
X = rng.standard_normal([m, n])

def test_sqrt():
   num = 25
   assert np.sqrt(num) == 5
