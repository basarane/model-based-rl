from keras import backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces

class DqnRMSprop(Optimizer):
	"""RMSProp optimizer (DQN Variant).
	https://arxiv.org/pdf/1308.0850v5.pdf p.23
	It is recommended to leave the parameters of this optimizer
	at their default values
	(except the learning rate, which can be freely tuned).
	This optimizer is usually a good choice for recurrent
	neural networks.
	# Arguments
		lr: float >= 0. Learning rate.
		rho1: float >= 0.
		rho2: float >= 0.
		epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
		decay: float >= 0. Learning rate decay over each update.
	# References
		- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
	"""

	def __init__(self, lr=0.001, rho1=0.95, rho2=0.95, epsilon=0.01, decay=0., print_layer=-1,
				 **kwargs):
		super(DqnRMSprop, self).__init__(**kwargs)
		with K.name_scope(self.__class__.__name__):
			self.lr = K.variable(lr, name='lr')
			self.rho1 = K.variable(rho1, name='rho1')
			self.rho2 = K.variable(rho2, name='rho2')
			self.decay = K.variable(decay, name='decay')
			self.iterations = K.variable(0, dtype='int64', name='iterations')
		if epsilon is None:
			epsilon = K.epsilon()
		self.epsilon = epsilon
		self.initial_decay = decay
		self.print_layer = print_layer

	@interfaces.legacy_get_updates_support
	def get_updates(self, loss, params):
		grads = self.get_gradients(loss, params)
		if self.print_layer >= 0:
			print(grads)
		
		accumulators1 = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
		accumulators2 = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
		self.weights = accumulators1
		self.updates = [K.update_add(self.iterations, 1)]

		lr = self.lr
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * K.cast(self.iterations,
												  K.dtype(self.decay))))

		I = 0
		for p, g, a1, a2 in zip(params, grads, accumulators1, accumulators2):
			# update accumulator
			#g = g * 48
			new_a1 = self.rho1 * a1 + (1. - self.rho1) * g
			new_a2 = self.rho2 * a2 + (1. - self.rho2) * K.square(g)
			self.updates.append(K.update(a1, new_a1))
			self.updates.append(K.update(a2, new_a2))
			tmp = K.sqrt(new_a2 - K.square(new_a1) + self.epsilon)
			deltas = lr * g / tmp
			new_p = p - deltas

			# Apply constraints.
			if getattr(p, 'constraint', None) is not None:
				new_p = p.constraint(new_p)
			self.updates.append(K.update(p, new_p))
			if I == self.print_layer:
				self.updates.append(K.print_tensor(g, message='dw: '))
				self.updates.append(K.print_tensor(deltas, 'Delta: '))
				self.updates.append(K.print_tensor(new_a1, 'g: '))
				self.updates.append(K.print_tensor(new_a2, 'g2: '))
				self.updates.append(K.print_tensor(tmp, 'tmp: '))
				self.updates.append(K.print_tensor(lr, 'lr: '))
			I += 1
		return self.updates

	def get_config(self):
		config = {'lr': float(K.get_value(self.lr)),
				  'rho1': float(K.get_value(self.rho1)),
				  'rho2': float(K.get_value(self.rho2)),
				  'decay': float(K.get_value(self.decay)),
				  'epsilon': self.epsilon}
		base_config = super(MyRMSprop, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

