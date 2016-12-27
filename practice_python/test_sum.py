import numpy as np
import tensorflow as tf

all_predictions = np.array([1,0])
y_test = [1,1]
#sum(all_predictions == y_test)
correct_predictions = float(sum(all_predictions == y_test))
#correct_predictions = tf.equal(all_predictions, y_test)
#print(correct_predictions)

#print(np.column_stack((all_predictions, y_test, [[1,2], [3,4]])))

c = [[None,None]]
a = [[1],[1]]
b = [[2,2], [3,3]]
print(np.concatenate([c, b]))


if 'asdf' in locals():
  print("Hi")


a = input()
print(a)
