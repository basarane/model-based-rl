import numpy as np
import keras.backend as K

if K.backend() == 'tensorflow':
	import tensorflow as tf
elif K.backend() == 'theano':
	from theano import tensor as T

def huber_loss(y_true, y_pred):
	clip_value = 1
	# Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
	# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
	# for details.
	assert clip_value > 0.

	coef = 1
	x = y_true - y_pred
	if np.isinf(clip_value):
		# Spacial case for infinity since Tensorflow does have problems
		# if we compare `K.abs(x) < np.inf`.
		return coef * K.sum(.5 * K.square(x))

	condition = K.abs(x) < clip_value
	#squared_loss = .5 * K.square(x)
	#linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
	squared_loss = 0.5 * K.square(x)
	linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
	if K.backend() == 'tensorflow':
		if hasattr(tf, 'select'):
			return coef * K.sum(tf.select(condition, squared_loss, linear_loss))  # condition, true, false
		else:
			return coef * K.sum(tf.where(condition, squared_loss, linear_loss))  # condition, true, false
	elif K.backend() == 'theano':
		return T.switch(condition, squared_loss, linear_loss)
	else:
		raise RuntimeError('Unknown backend "{}".'.format(K.backend()))