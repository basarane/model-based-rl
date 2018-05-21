from runner.runner import *
from keras.layers import Input, Dense, Concatenate, Add, Subtract, Lambda, Multiply
from keras import Model
from keras.optimizers import Adam, RMSprop
import numpy as np 
import keras
import keras.backend as K

from utils.summary_writer import SummaryWriter

class EnvModel(object):
	def __init__(self, ops):
		self.ops = ops
		pass
	def predict_next(self, current_state, action):
		pass
	def train_next(self, current_state, action):
		pass

class EnvOps(object):
	def __init__(self, input_size, action_count, learning_rate, mode = "test"):
		self.INPUT_SIZE = input_size
		self.ACTION_COUNT = action_count
		self.LEARNING_RATE = learning_rate
		self.mode = mode
		
class EnvModelCartpole(EnvModel):
	def __init__(self, ops):
		super(EnvModelCartpole, self).__init__(ops)
		self.model = self.get_model()
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
		x = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
		action_outputs = []
		done_outputs = []
		#reward_outputs = []
		reward_output = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
		losses = []
		loss_weights = []
		for I in range(self.ops.ACTION_COUNT):
			action_output = Dense(input_shape[0], kernel_initializer='he_uniform')(x)
			action_outputs.append(action_output)
			done_output = Dense(1, kernel_initializer='he_uniform', activation='sigmoid')(x)
			done_outputs.append(done_output)
			losses.append('mse')
			loss_weights.append(1)
			#reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
			#reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
			#reward_outputs.append(reward_output)
		losses.append('mse')
		loss_weights.append(1)
		for I in range(self.ops.ACTION_COUNT):
			losses.append('binary_crossentropy')
			loss_weights.append(0.3)
		model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		#my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
		model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
		return model
	def predict_next(self, current_state):
		return self.model.predict(np.array(current_state, dtype='f'))
	def train_next(self, current_state, next_states):
		return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)

class EnvModelLine(EnvModel):
	def __init__(self, ops):
		super(EnvModelLine, self).__init__(ops)
		self.model = self.get_model()
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(24,activation="relu", kernel_initializer='he_uniform')(x)
		action_outputs = []
		done_outputs = []
		#reward_outputs = []
		reward_output = Dense(self.ops.ACTION_COUNT, kernel_initializer='he_uniform')(x)
		losses = []
		loss_weights = []
		for I in range(self.ops.ACTION_COUNT):
			action_output = Dense(input_shape[0], kernel_initializer='he_uniform')(x)
			action_outputs.append(action_output)
			done_output = Dense(1, kernel_initializer='he_uniform', activation='sigmoid')(x)
			done_outputs.append(done_output)
			losses.append('mse')
			loss_weights.append(1)
			#reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
			#reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
			#reward_outputs.append(reward_output)
		losses.append('mse')
		loss_weights.append(1)
		for I in range(self.ops.ACTION_COUNT):
			losses.append('binary_crossentropy')
			loss_weights.append(0.3)
		model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		#my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
		model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
		return model
	def predict_next(self, current_state):
		return self.model.predict(np.array(current_state, dtype='f'))
	def train_next(self, current_state, next_states):
		return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)	
		
class EnvModelLunarLander(EnvModel):
	def __init__(self, ops):
		super(EnvModelLunarLander, self).__init__(ops)
		self.model = self.get_model()
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(256,activation="relu")(x)
		x = Dense(256,activation="relu")(x)
		action_outputs = []
		done_outputs = []
		#reward_outputs = []
		c = Dense(256,activation="relu")(x)
		reward_output = Dense(self.ops.ACTION_COUNT)(c)
		losses = []
		loss_weights = []
		for I in range(self.ops.ACTION_COUNT):
			a = Dense(256,activation="relu")(x)
			action_output = Dense(input_shape[0])(a)
			action_outputs.append(action_output)
			b = Dense(256,activation="relu")(x)
			done_output = Dense(1, activation='sigmoid')(b)
			done_outputs.append(done_output)
			losses.append('mse')
			loss_weights.append(1) # next state loss weight
			#reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
			#reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
			#reward_outputs.append(reward_output)
		losses.append('mse')
		loss_weights.append(0.3) #reward loss weight
		for I in range(self.ops.ACTION_COUNT):
			losses.append('binary_crossentropy')
			loss_weights.append(0.3) #done loss weight
		model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		#my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
		model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
		return model
	def predict_next(self, current_state):
		return self.model.predict(np.array(current_state, dtype='f'))
	def train_next(self, current_state, next_states):
		return self.model.train_on_batch(np.array(current_state, dtype='f'), next_states)
		
class EnvModelLunarLander2(EnvModelLunarLander):
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		rx = input
		ax = input
		dx = input
		action_outputs = []
		done_outputs = []
		#reward_outputs = []
		rx = Dense(1024,activation="sigmoid")(rx)
		rx = Dense(1024,activation="sigmoid")(rx)
		reward_output = Dense(self.ops.ACTION_COUNT)(rx)
		losses = []
		loss_weights = []
		for I in range(self.ops.ACTION_COUNT):
			ax = Dense(1024,activation="sigmoid")(ax)
			ax = Dense(1024,activation="sigmoid")(ax)
			action_output = Dense(input_shape[0])(ax)
			action_outputs.append(action_output)
			dx = Dense(1024,activation="sigmoid")(dx)
			dx = Dense(1024,activation="sigmoid")(dx)
			done_output = Dense(1, activation='sigmoid')(dx)
			done_outputs.append(done_output)
			losses.append('mse')
			loss_weights.append(1) # next state loss weight
			#reward_hidden = Dense(256,activation="relu", kernel_initializer='he_uniform')(x)
			#reward_output = Dense(1, kernel_initializer=keras.initializers.random_uniform(-0.02, 0.02))(reward_hidden)
			#reward_outputs.append(reward_output)
		losses.append('mse')
		loss_weights.append(0.3) #reward loss weight
		for I in range(self.ops.ACTION_COUNT):
			losses.append('binary_crossentropy')
			loss_weights.append(0.3) #done loss weight
		model = Model(inputs=[input], outputs=action_outputs + [reward_output] + done_outputs)
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		#my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE, rho=0.90, decay=0.0) #epsilon=None, 
		model.compile(optimizer=my_optimizer,loss=losses, loss_weights=loss_weights)
		return model
		
class EnvLearner(RunnerListener):
	def __init__(self, sampler, model, sw = None, reward_scale = 1):
		super(RunnerListener, self).__init__()
		self.sampler = sampler
		self.model = model
		label_action_losses = ['loss_action_' + str(a+1) for a in range(self.model.ops.ACTION_COUNT)]
		label_action_dones = ['loss_done_' + str(a+1) for a in range(self.model.ops.ACTION_COUNT)]
		self.sw = SummaryWriter(sw, ['Total loss'] + label_action_losses + ['loss_r'] + label_action_dones)
		self.total_step_count = 0
		self.reward_scale = reward_scale
	def on_step(self, ob, action, next_ob, reward, done):
		self.total_step_count += 1
		if self.sampler.has_sample():
			samples = self.sampler.get_sample()
			current_states = [a['current_state'] for a in samples]
			next_states = [a['next_state'] for a in samples]
			actions = [a['action'] for a in samples]
			rewards = [a['reward'] for a in samples]
			dones = [1 if a['done'] else 0 for a in samples]
			#@ersin - unutma, asagidaki satir  cartpole icin eklenmis bir kod sadece
			#dones = [1 if a['reward']<0 else 0 for a in samples]
			est_next_states = self.model.predict_next(current_states)
			#print(len(est_next_states))
			ac = (len(est_next_states)-1)/2
			for I in range(len(actions)):
				#print(I, actions[I], next_states[I])
				est_next_states[actions[I]][I] = next_states[I]
				r = min(max(rewards[I], -1), 1)
				est_next_states[ac][I] = r #rewards[I] * self.reward_scale
				est_next_states[ac + 1 + actions[I]][I] = dones[I]
				#est_next_states[actions[I]+ac][I] = 0
			#print(est_next_states)
			loss = self.model.train_next(current_states, est_next_states)
			self.sw.add(loss, self.total_step_count)
			
		#if reward < 0:
		#	print('onstep', ob, action, next_ob, reward, done)
		pass

def output_of_lambda(input_shape):
    return (input_shape[0], 1)

def my_max(x):
    return K.max(x, axis=-1, keepdims=False)

class VNetwork(object):
	def __init__(self, ops):
		self.ops = ops
		self.model = self.get_model()
	def get_model(self):
		raise NotImplementedException()
	def v_value(self, x):
		return self.model.predict_on_batch(x)
	def get_weights(self):
		return self.model.get_weights()
	def set_weights(self, w):
		self.model.set_weights(w)
	def clone(self):
		raise NotImplementedException()
		
class CartPoleVNetwork(VNetwork):
	def __init__(self, ops):
		super(CartPoleVNetwork, self).__init__(ops)
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
		x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
		v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
		model = Model(inputs=[input], outputs=[v])
		my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse')
		return model
	def clone(self):
		new_model = CartPoleVNetwork(self.ops)
		new_model.set_weights(self.get_weights())
		return new_model
	
class LineVNetwork(VNetwork):
	def __init__(self, ops):
		super(LineVNetwork, self).__init__(ops)
	def get_model(self):
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
		v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
		model = Model(inputs=[input], outputs=[v])
		my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
		model.compile(optimizer=my_optimizer,loss='mse')
		return model
	def clone(self):
		new_model = LineVNetwork(self.ops)
		new_model.set_weights(self.get_weights())
		return new_model	
	
class TDNetwork(object):
	def __init__(self, env_model, v_model, env_ops):
		self.env_model = env_model
		self.ops = env_ops
		self.v_model = v_model
		self.v_model_eval = v_model.clone()
		self.td_model = self.get_model()
	def get_model(self):
		self.env_model.trainable = False
		input_shape=self.ops.INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		#x = input
		#x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
		#x = Dense(24,activation="relu")(x) #, kernel_initializer='he_uniform'
		#v = Dense(1)(x) #activation="relu", , kernel_initializer='he_uniform'
		#v_model = Model(inputs=[input], outputs=[v])
		v = self.v_model.model(input)
		env_output = self.env_model(input)
		next_v = []
		for I in range(self.ops.ACTION_COUNT):
			#@ersin - use different eval network as in DQN
			#one_v = self.v_model_eval.model(env_output[I])
			one_v = self.v_model.model(env_output[I])
			one_v.trainable = False
			next_v.append(one_v)
		next_v_tensor = Concatenate()(next_v)
		next_v_discounted_tensor = Lambda(lambda x: x * 0.99)(next_v_tensor)
		done_tensor = Concatenate()(env_output[self.ops.ACTION_COUNT+1:])
		not_done_tensor = Lambda(lambda x: 1 - x)(done_tensor)
		done_fix = Multiply()([next_v_discounted_tensor, not_done_tensor])
		reward = env_output[self.ops.ACTION_COUNT]
		#@ersin - *100 u yine CartPole icin eklenmisim sanirim
		#reward = Lambda(lambda x: K.switch(x < 0, x*100, x))(reward)
		#reward = Lambda(lambda x: K.print_tensor(x, message='reward'))(reward)
		est_v = Add()([reward, done_fix])
		est_max_v = Lambda(my_max, output_shape=output_of_lambda)(est_v)
		# disable learning of value func within max
		#est_max_v.trainable = False
		td_error = Subtract()([v, est_max_v])
		td_model = Model(inputs=[input], outputs=[td_error])
		my_optimizer = Adam(lr=self.ops.LEARNING_RATE)
		#my_optimizer = RMSprop(lr=self.ops.LEARNING_RATE)
		td_model.compile(optimizer=my_optimizer,loss='mse')
		return td_model
	def train(self, state):
		return self.td_model.train_on_batch(state, np.zeros((len(state), 1), dtype='f'))
