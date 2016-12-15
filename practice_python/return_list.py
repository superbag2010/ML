#! /usr/bin/python3

import numpy as np
import sys

def return_list(x):
  a = 1*x
  b = 2*x
  return {a : b}


a = {}
a.update(return_list(1))
a.update(return_list(3))
a.update(return_list(2))
print("a = {}, b = ".format(a))

b = 3
print(b)
sys.exit(3)
