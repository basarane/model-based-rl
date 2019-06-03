import tensorflow.keras
# the default initializer used by torch (as original dqn code uses this)
def dqn_uniform(seed=None):  
    return keras.initializers.VarianceScaling(scale=0.3333333,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)
