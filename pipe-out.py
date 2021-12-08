#!/usr/bin/env python

import time
import sys
import numpy as np
import math

upper = 24 * 32
nums = np.zeros((24 * 32,))
nums = np.reshape(nums, (24,32))

# не забудь перевернуть стороны
# np.fliplr(data_array)

temp = 20
for i in range(3, 6):
  for j in range(3, 6):
    nums[i][j] = temp
  temp = temp + 2

temp = 30
for i in range(10, 19):
  for j in range(10, 19):
    nums[i][j] = temp
  temp = temp + math.copysign(2, 14 - i)

nums = np.reshape(nums, (nums.shape[0] * nums.shape[1],))

# print(nums)
while True:
  sys.stdout.write(" ".join(np.char.mod('%03d', nums)) + '\n')
  sys.stdout.flush()
  time.sleep(1)
