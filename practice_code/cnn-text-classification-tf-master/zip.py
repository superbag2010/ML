import numpy as np

x = np.array([ [[1,1,1]], [[4,4,4]] ])
y = np.array([ [11], [33] ])
batches = list(zip(x, y))
print("batches = {}".format(batches))

batches = np.array(batches)
print("x.shape = {}, y.shape = {}".format(x.shape, y.shape))
print("batches = {}".format(batches))
"""
print("batches = {}".format(batches))
for batch in batches:
  x_batch, y_batch = zip(*batch)
  print("batch = {}".format(batch))
  print("*batch = {}".format(*batch))
#  print("x_batch = {}, y_batch = {}".format(x_batch, y_batch))
"""

"""
print("")

batches = list(zip([7,7,7], [11,12,13]))

print("batches = {}".format(batches))
print("======================================")
for batch in batches:
  x_batch, y_batch = zip(*batch)
  print("batch = {}".format(batch))
  print("x_batch = {}, y_batch = {}".format(x_batch,y_batch))
  print("======================================")
"""
