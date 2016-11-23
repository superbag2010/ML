import tensorflow as tf

arr = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5]])
# argmax(arr, 0) = [1 1 1 1],  argmax(arr, 1) = [3 3]

#arr = tf.constant([2, 3, 4, 5])   # argmax(arr, 0) = 3
argmax = tf.argmax(arr, 0)

print(vars(arr))
with tf.Session() as sess:
    print(sess.run(arr))
    print(sess.run(argmax))


""" tf.eval()
sess = tf.Session()
with sess.as_default():
    print(argmax.eval())
"""
