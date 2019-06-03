import random 
from tensorflow.keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np
from tensorflow.keras.layers import Lambda

import tensorflow as tf
def printLayer(x, message):
	#return Lambda(lambda x: K.print_tensor(x, message))(x)
	return Lambda(lambda x: tf.Print(x, [x], message=message, summarize=1000, first_n = 2))(x)

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows 
          are taken as centers)
    """
    def __init__(self, X):
        self.X = X 

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx,:]
