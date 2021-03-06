from . initializers import dqn_uniform
from tensorflow.keras.layers import Input, Permute, ZeroPadding2D, Conv2D, Flatten, Dense, Add, Subtract, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from . optimizers import DqnRMSprop
from . loss import huber_loss, huber_loss_mse
import numpy as np
import tensorflow.keras.backend as K
from nets.layers import printLayer

def init_nn_library(use_gpu = True, gpu_id = "0", memory_fraction = 0.1):
	if use_gpu:
		import tensorflow as tf
		#from tensorflow.keras.backend.tensorflow_backend import set_session
		from tensorflow.keras.backend import set_session
		config = tf.ConfigProto(log_device_placement=False)
		config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
		config.gpu_options.allow_growth = True
		print(f'Using gpu {gpu_id}')
		if not gpu_id == 'any':
			print('Using specific gpu')
			config.gpu_options.visible_device_list = gpu_id
		set_session(tf.Session(config=config))
	else:
		import os
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
	
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
	def set_weights(self, val):
		raise NotImplementedException()
	
class TabularQModel(QModel):
	def __init__(self, ops = None, model = None):
		super(TabularQModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		return np.random.uniform(-0.1,0.1,(self.ops.INPUT_SIZE, self.ops.ACTION_COUNT))
	def q_value(self, state):
		if type(state) is list:
			return np.array([self.model[s[0] if type(s) is list else s,:] for s in state])
		return self.model[state,:]
	def q_update(self, state, target):
		loss = 0
		if type(state) is list:
			td_total = 0
			for I in range(len(state)):
				s = state[I]
				t = target[I]
				td = t - self.model[s[0],:]
				self.model_update[s[0],:] = self.model_update[s[0],:] + self.ops.LEARNING_RATE*td
				td_total += td
			loss = td_total
		else:
			td = target - self.model[state,:]
			self.model_update[state,:] = self.model_update[state,:] + self.ops.LEARNING_RATE*td
			loss = td
		if self.model_update is not self.model:
			self.model = np.copy(self.model_update)
		return loss
	def clone_model(self):
		#m = self.get_model()
		m = self.get_weights()
		return m
	def clone(self):
		m = self.clone_model()
		return TabularQModel(self.ops, m)
	def get_weights(self):
		return np.copy(self.model)
	def set_weights(self, val):
		self.model = np.copy(val)
		
def my_mean(x, ACTION_COUNT):
	x = K.mean(x, axis=1, keepdims=True)
	x = K.tile(x, (1,ACTION_COUNT))
	return x
	
class DQNModel(QModel):
	def __init__(self, ops = None, model = None):
		super(DQNModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		print('*************GET MODEL DQN ***********************')
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
		#model.summary()
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
		loss = self.model_update.train_on_batch(state, target)
		if self.model_update is not self.model:
			self.model.set_weights(self.model_update.get_weights())
		return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return DQNModel(self.ops, m)

		
#class A3CModel(DQNModel):
# renamed
class DQNModel_AsyncQ(DQNModel):
	def __init__(self, ops = None, model = None):
		super(DQNModel_AsyncQ, self).__init__(ops, model)
	def get_model(self):
		print('*************GET MODEL DQNModel_AsyncQ ***********************')
		input_shape=(self.ops.AGENT_HISTORY_LENGTH,) + self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = Permute((2,3,1))(input)
		x = Conv2D(filters=16,kernel_size=8,strides=4,padding="valid",activation="relu", kernel_initializer=dqn_uniform(), name='layer1')(x)
		x = Conv2D(filters=32,kernel_size=4,strides=2,padding="valid",activation="relu", kernel_initializer=dqn_uniform(), name='layer2')(x)
		x = Flatten()(x)
		if not self.ops.dueling_network:
			x = Dense(256,activation="relu", kernel_initializer=dqn_uniform())(x)
			y = Dense(self.ops.ACTION_COUNT, kernel_initializer=dqn_uniform())(x)
		else:
			xv = Dense(256,activation="relu", kernel_initializer=dqn_uniform(), name="dense_v")(x)
			xa = Dense(256,activation="relu", kernel_initializer=dqn_uniform(), name="dense_a")(x)
			v = Dense(1, kernel_initializer=dqn_uniform(), name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer=dqn_uniform(), name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		model._make_predict_function()
		#model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		#my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.99, epsilon=0.01)
		model.compile(optimizer=my_optimizer,loss=huber_loss) #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
		
class ActorCriticModel(object):
	def __init__(self, ops = None, model = None):
		#super(ActorCriticModel, self).__init__(ops, model)
		self.ops = ops
		if model is None:
			self.model_actor, self.model_critic, self.model = self.get_model()
		else:
			self.model_actor, self.model_critic, self.model = model
	def get_network(self):
		raise NotImplementedException()
	def get_optimizer(self):
		return Adam(lr=0.001)
	def get_model(self):
		print('*************GET MODEL ActorCriticModel *********************** ' + str(self.ops.INPUT_SIZE))
		input, pi, V = self.get_network()
		
		def actor_optimizer():
			a_t = K.placeholder(shape=[None, self.ops.ACTION_COUNT])
			A = K.placeholder(shape=(None, ))
			policy = model_actor.output
			Lpi = -K.sum(K.log(K.sum(policy*a_t, axis=1) + 1e-10) * K.stop_gradient(A))
			LH = K.sum(K.sum(pi * K.log(policy + 1e-10), axis=1))
			L = Lpi + 0.01 * LH
			
			optimizer = self.get_optimizer()
			updates = optimizer.get_updates(model_actor.trainable_weights, [], L)
			train = K.function([model_actor.input, a_t, A], [L], updates=updates)
			return train
		def critic_optimizer():
			R = K.placeholder(shape=(None,))
			critic = model_critic.output
			#critic = K.print_tensor(critic, message='critic: ')
			#critic = printLayer(critic,'critic_output')
			Lv = K.mean(K.square(R - critic))
			Lv = K.sum(Lv)
			#Lv = printLayer(Lv,'loss')

			#Lv = K.print_tensor(Lv, message='Lv: ')
			optimizer = self.get_optimizer()
			updates = optimizer.get_updates(model_critic.trainable_weights, [], Lv)
			train = K.function([model_critic.input, R], [Lv], updates=updates)
			return train		
		model_actor = Model(inputs=[input], outputs=[pi])
		model_critic = Model(inputs=[input], outputs=[V])
		model = Model(inputs=[input], outputs=[pi, V])
		model_actor._make_predict_function()
		model_critic._make_predict_function()
		model._make_predict_function()
		model_actor.manual_optimizer = actor_optimizer()
		model_critic.manual_optimizer = critic_optimizer()
		print('Actor Model')
		print(model_actor.summary())
		print('Critic Model')
		print(model_critic.summary())
		print('Complete Model')
		print(model.summary())
		return model_actor, model_critic, model
	# bu methodlar degisecek
	#def q_value(self, state):
	#	state = np.array(state, dtype='f')/255.0
	#	return self.model.predict_on_batch(state)
	#def q_update(self, state, target):
	#	state = np.array(state, dtype='f')/255.0
	#	loss = self.model_update.train_on_batch(state, target)
	#	if self.model_update is not self.model:
	#		self.model.set_weights(self.model_update.get_weights())
	#	return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone_model(self):
		m = self.get_model()
		m[2].set_weights(self.get_weights())
		return m
	def clone(self):
		m = self.clone_model()
		return ActorCriticModel(self.ops, m)
		
class CartPoleActorCriticModel(ActorCriticModel):
	def __init__(self, ops = None, model = None):
		super(CartPoleActorCriticModel, self).__init__(ops, model)
	def get_network(self):
		# @ersin - Atari'de AGENT_HISTORY_LENGTH eklenmeli
		input_shape = self.ops.INPUT_SIZE  #+ (self.ops.AGENT_HISTORY_LENGTH,)
		input = Input(shape=input_shape, name='observation')
		x = input
		#x = Conv2D(filters=16,kernel_size=8,strides=4,padding="valid",activation="relu", name='layer1')(x)
		#x = Conv2D(filters=32,kernel_size=4,strides=2,padding="valid",activation="relu", name='layer2')(x)
		##x = Permute((3,1,2))(x)
		#x = Flatten()(x)
		#x = Dense(256,activation="relu")(x)
		x = Dense(24,activation="relu", kernel_initializer='glorot_uniform')(x)
		#x = Dense(24,activation="relu")(x)
		pi_hidden = Dense(24,activation="relu", kernel_initializer='glorot_uniform')(x)
		V_hidden = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
		pi = Dense(self.ops.ACTION_COUNT,activation="softmax", kernel_initializer='glorot_uniform')(pi_hidden)
		V = Dense(1, kernel_initializer='he_uniform')(V_hidden)
		return input, pi, V
	def clone(self):
		m = self.clone_model()
		return CartPoleActorCriticModel(self.ops, m)
	
class AtariActorCriticModel(ActorCriticModel):
	def __init__(self, ops = None, model = None):
		super(AtariActorCriticModel, self).__init__(ops, model)
	def get_network(self):
		input_shape = (self.ops.AGENT_HISTORY_LENGTH,) + self.ops.INPUT_SIZE 
		input = Input(shape=input_shape, name='observation')
		x = Permute((2,3,1))(input)
		x = Conv2D(filters=16,kernel_size=8,strides=4,padding="valid",activation="relu", name='layer1')(x)
		x = Conv2D(filters=32,kernel_size=4,strides=2,padding="valid",activation="relu", name='layer2')(x)
		x = Flatten()(x)
		x = Dense(256,activation="relu")(x)
		pi = Dense(self.ops.ACTION_COUNT,activation="softmax")(x)
		V = Dense(1)(x)
		return input, pi, V
	def get_optimizer(self):
		return RMSprop(lr=0.00025, rho=0.99, epsilon=0.01)
	def clone(self):
		m = self.clone_model()
		return AtariActorCriticModel(self.ops, m)		

	
class CartPoleModel(QModel):
	def __init__(self, ops = None, model = None):
		super(CartPoleModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		print('*************GET MODEL DQN ***********************')
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
		if not self.ops.dueling_network:
			x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
			y = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
		else:
			xv = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_v")(x)
			xa = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_a")(x)
			v = Dense(1, kernel_initializer='he_uniform', name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform', name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		#model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		#my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		#my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse') #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
	def q_value(self, state):
		state = np.array(state, dtype='f')
		return self.model.predict_on_batch(state)
	def q_update(self, state, target):
		state = np.array(state, dtype='f')
		loss = self.model_update.train_on_batch(state, target)
		if self.model_update is not self.model:
			self.model.set_weights(self.model_update.get_weights())
		return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return CartPoleModel(self.ops, m)

class LunarLanderModel(QModel):
	def __init__(self, ops = None, model = None):
		super(LunarLanderModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		print('*************GET MODEL DQN ***********************')
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(512,activation="relu", kernel_initializer='he_uniform')(x)
		if not self.ops.dueling_network:
			x = Dense(512,activation="relu", kernel_initializer='he_uniform')(x)
			y = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
		else:
			xv = Dense(512,activation="relu", kernel_initializer='he_uniform', name="dense_v")(x)
			xa = Dense(512,activation="relu", kernel_initializer='he_uniform', name="dense_a")(x)
			v = Dense(1, kernel_initializer='he_uniform', name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform', name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		#model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		#my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		#my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse') #
		#model.compile(optimizer=my_optimizer,loss=huber_loss) #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
	def q_value(self, state):
		state = np.array(state, dtype='f')
		return self.model.predict_on_batch(state)
	def q_update(self, state, target):
		state = np.array(state, dtype='f')
		loss = self.model_update.train_on_batch(state, target)
		if self.model_update is not self.model:
			self.model.set_weights(self.model_update.get_weights())
		return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return CartPoleModel(self.ops, m)
				
class LineModel(QModel):
	def __init__(self, ops = None, model = None):
		super(LineModel, self).__init__(ops, model)
		self.model_update = self.model
	def get_model(self):
		print('*************GET MODEL DQN LINE ***********************')
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		#x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
		if not self.ops.dueling_network:
			x = Dense(10,activation="relu")(x)
			y = Dense(self.ops.ACTION_COUNT)(x)
		else:
			xv = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_v")(x)
			xa = Dense(24,activation="relu", kernel_initializer='he_uniform', name="dense_a")(x)
			v = Dense(1, kernel_initializer='he_uniform', name="v")(xv) #,activation="relu"
			a = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform', name="a")(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': self.ops.ACTION_COUNT}, name="mean_a")(a)
			y1 = Add(name="v_plus_a")([v, a])
			y = Subtract(name="q_value")([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		#model.summary()
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		#my_optimizer = DqnRMSprop(lr=self.ops.LEARNING_RATE, rho1=self.ops.GRADIENT_MOMENTUM, rho2=self.ops.SQUARED_GRADIENT_MOMENTUM, epsilon=self.ops.MIN_SQUARED_GRADIENT, print_layer=-1)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse') #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
		return model
	def q_value(self, state):
		state = np.array(state, dtype='f')
		return self.model.predict_on_batch(state)
	def q_update(self, state, target):
		state = np.array(state, dtype='f')
		loss = self.model_update.train_on_batch(state, target)
		if self.model_update is not self.model:
			self.model.set_weights(self.model_update.get_weights())
		return loss
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		m = self.clone_model()
		return LineModel(self.ops, m)
				