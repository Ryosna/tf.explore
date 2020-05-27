import tensorflow as tf 
import numpy

# rank num = num of nested variables in an array 
meow = tf.Variable([["meow"],["roof"]], tf.string)
print(tf.rank(meow))