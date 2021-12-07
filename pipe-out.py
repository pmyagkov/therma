#!/usr/bin/env python

import time
import sys
import numpy as np

upper = 24 * 32
nums = np.arange(0, upper)# np.zeros((upper,))
while True:
  nums = np.random.rand(upper) * 20
  sys.stdout.write(" ".join(np.char.mod('%03d', nums)) + '\n')
  sys.stdout.flush()
  time.sleep(1)
