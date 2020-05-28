import tensorflow as tf 
import pandas as pd
import numpy
import matplotlib.pyplot as plt

"""
# rank num = num of nested arrays in an array 
meow = tf.Variable([["meow"],["roof"]], tf.string)
print(tf.rank(meow))
print(meow.shape)
rank2Tensor = tf.Variable([["Okay","Yes", "Maybe"],["Okay", "No", "Sure"],["Okay", "Nah", "Why Not"]], tf.string)
print(tf.rank(rank2Tensor))
print(rank2Tensor.shape)

# EX) shape [1,2,3] means 1 list, 2 nested lists, and 3 elements inside each nested list. In total, there are 6 elements. 
tensor1 = tf.ones([1,2,3])
print(tensor1)

#reshaping tensor1 to have 2 lists, 3 nested lists, and 1 element inside each ensted list. Still 6 elements in total. 
tensor2 = tf.reshape(tensor1, [2,3,1])
print(tensor2)

#reshaping tensor1 to have 3 lists. The -1 allows for the tensor to calculate the size automatically
tensor3 = tf.reshape(tensor1, [3,-1])
print(tensor3)
"""

# t = tf.zeros([5,5,5,5])
# print(t)

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
# print(dftrain.head()) # .head() allows you to look at the data
y_train = dftrain.pop('survived') #takes the "survived" column and removes/pops it out of the dataset 
#the reason it's in a variable is because pop returns what is removed. Now the survived column is in the y_train variable. 
y_eval = dfeval.pop('survived')
# print(dftrain.loc[0], y_train.loc[0]) #locating row 0 
# print(dftrain.shape) #prints (627,9) meaning it has 627 rows and 9 collumns of data
