from initializers import dqn_uniform
from keras.layers import Input, Permute, ZeroPadding2D, Conv2D, Flatten, Dense, Add, Subtract, Lambda
from keras import Model
from optimizers import DqnRMSprop
from loss import huber_loss
import numpy as np
import keras.backend as K

def init_nn_library(use_gpu = True, gpu_id = "0", memory_fraction = 0.3):
	if use_gpu:
		import tensorflow as tf
		from keras.backend.tensorflow_backend import set_session
		config = tf.ConfigProto(log_device_placement=False)
		config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
		config.gpu_options.allow_growth = True
		config.gpu_options.visible_device_list = gpu_id
		set_session(tf.Session(config=config))

class DqnOps(object):
	def __init__(self, action_count):
		self.AGENT_HISTORY_LENGTH = 4
		self.INPUT_SIZE = (84,84)
		self.dueling_network = False
		self.ACTION_COUNT = action_count
		self.LEARNING_RATE = 0.00025
		self.GRADIENT_MOMENTUM=0.95
		self.SQUARED_GRADIENT_MOMENTUM=0.95
		self.MIN_SQUARED_GRADIENT=0.01

class QModel(object):
	def __init__(self, ops = None, model = None):
		self.ops = ops
		if model is None:
			self.model = self.get_model()
		else:
			self.model = model
	def get_model(self):
		raise NotImplementedException()
	def q_value(self, state):
		raise NotImplementedException()
	def q_update(self, state, target):
		raise NotImplementedException()
	def clone_model(self):
		m = self.get_model()
		m.set_weights(self.get_weights())
		return m
	def clone(self):
		raise NotImplementedException()
	def get_weights(self):
		raise NotImplementedException()
	def set_weights(self):
		raise NotImplementedException()
	
	
def my_mean(x, ACTION_COUNT):
	x = K.mean(x, axis=1, keepdims=True)
	x = K.tile(x, (1,ACTION_COUNT))
	return x
	
class DQNModel(QModel):
	def __init__(self, ops = None, model = None):
		super(DQNModel, self).__init__(ops, model)
	def get_model(self):
		input_shape=(self.ops.AGENT_HISTORY_LENGTH,) + self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = Permute((2,3,1))(input)
		x = ZeroPadding2D(padding=((1,0),(1,0)), name='layer1_padding')(x)
		x = Conv2D(filters=32,kernel_size=8,strides=4,padding="valid",activation="relu", kernel_initializer=dqn_uniform(), name='layer1')(x)
		x = Conv2D(filters=64,kernel_size=4,strides=2,padding="valid",activation="relu", kernel_initializer=dqn_uniform(), name='layer2')(x)
		x = Conv2D(filters=64,kernel_size=3,strides=1,padding="valid",activation="relu", kernel_initializer=dqn_uniform(), name='layer3')(x)
		x = Permute((3,1,2))(x)
		x = Flatten()(x)
		if not self.ops.dueling_network:
			x = Dense(512,activation="relu", kernel_initializer=dqn_uniform())(x)
			y = Dense(self.ops.ACTION_COUNT, kernel_initializer=dqn_uniform())(x)
		else:
			xv = Dense(512,activation="relu", kernel_initializer=dqn_uniform(), name="dense_v")(x)
			xa = Dense(512,activation="relu", kernel_initializer=dqn_uniform(), name="dense_a")(x)
			v = Dense(1, kernel_initializer=dqn_uniform(), name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer=dqn_uniform(), name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		model.compile(optimizer=my_optimizer,loss=huber_loss) #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
	def q_value(self, state):
		state = np.array(state, dtype='f')/255.0
		return self.model.predict_on_batch(state)
	def q_update(self, state, target):
		state = np.array(state, dtype='f')/255.0
		return self.model.train_on_batch(state, target)
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return DQNModel(self.ops, m)
	