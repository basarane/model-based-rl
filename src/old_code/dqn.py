from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='DQN train')
parser.add_argument('game', type=str, help='game name')
parser.add_argument('--output_dir', type=str, default=None, help='output directory')
parser.add_argument('--mode', type=str, default="train", help='mode (train or test)')
parser.add_argument('--enable-render', type=bool, default=False, help='enable render')
parser.add_argument('--render-step', type=int, default=4, help='Render at this steps')
parser.add_argument('--last-weightfile', nargs='?', type=str, default=None, help='last weight file')
parser.add_argument('--test-epsilon', nargs='?', type=float, default=0.05, help='test epsilon value')
parser.add_argument('--env-model', nargs='?', type=str, default=None, help='environment model type: env or acvp')
parser.add_argument('--env-weight', nargs='?', type=str, default=None, help='environment model weight file')
parser.add_argument('--env-mean-image', type=str, default=None, help='Load mean image')
parser.add_argument('--env-reward-weight', type=str, default=None, help='weight file for reward estimation')
parser.add_argument('--env-state-multiplier', type=float, default=1.0, help='state is multiplied by this value for reward estimation')
parser.add_argument('--gpu', type=str, default="1", help='Which gpu to use')
parser.add_argument('--REPLAY_MEMORY_SIZE', type=int, default=None, help='Replay memory size')
parser.add_argument('--REPLAY_START_SIZE', type=int, default=None, help='Replay start size')
parser.add_argument('--double-dqn', type=bool, default=False, help='Apply Double DQN algorithm')
parser.add_argument('--dueling-network', type=bool, default=False, help='Apply Dueling Network architecture')
parser.add_argument('--actor-critic', type=bool, default=False, help='Use actor-critic method')
parser.add_argument('--cpu', type=bool, default=False, help='Use CPU, invalidates gpu')
parser.add_argument('--torch-compare-path', type=str, default=None, help='The directory containing output of original dqn code') #../Human_Level_Control_through_Deep_Reinforcement_Learning_not_orig/dqn/output/


args = parser.parse_args()

# disable gpu, use cpu
if args.cpu:
	import os
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

if not args.output_dir is None:
	import os, sys
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

ENABLE_RENDER = args.enable_render

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Conv2D, ZeroPadding2D, Input, Add, Subtract, RepeatVector, Lambda
from keras.optimizers import Adam, RMSprop
from dqn_rmsprop import MyRMSprop
import keras.backend as K
from keras.utils import to_categorical

import keras
print('Keras version', keras.__version__)

import time
import gym
import numpy as np
from PIL import Image
from gym import wrappers
if ENABLE_RENDER:
	from gym.envs.classic_control import rendering
import sys
from memory import SequentialMemory
import random

from models import load_model
from models import my_predict
from utils_old import toRGBImage
from utils_old import prediction_to_image
from utils_old import draw_reward
from utils_old import get_obs_input

INPUT_SIZE=(84,84)
MINIBATCH_SIZE=32
REPLAY_MEMORY_SIZE=int(1e6)
AGENT_HISTORY_LENGTH=4
TARGET_NETWORK_UPDATE_FREQUENCY=1e4
DISCOUNT_FACTOR=0.99
ACTION_REPEAT=4
UPDATE_FREQUENCY=4
LEARNING_RATE=0.00025
GRADIENT_MOMENTUM=0.95
SQUARED_GRADIENT_MOMENTUM=0.95
MIN_SQUARED_GRADIENT=0.01
INITIAL_EXPLORATION=1
FINAL_EXPLORATION=0.1
FINAL_EXPLORATION_FRAME=1e6
REPLAY_START_SIZE=50000
NO_OP_MAX=30
MAX_REWARD=1
MIN_REWARD=-1
SAVE_FREQ=20000

if not args.REPLAY_MEMORY_SIZE is None:
	REPLAY_MEMORY_SIZE = args.REPLAY_MEMORY_SIZE

if not args.REPLAY_START_SIZE is None:
	REPLAY_START_SIZE = args.REPLAY_START_SIZE
	
print(REPLAY_MEMORY_SIZE, REPLAY_START_SIZE)

class RandomAgent(object):
	def __init__(self, action_space):
		self.action_space = action_space
	def act(self, observation, reward, done):
		return self.action_space.sample()

if ENABLE_RENDER:
	viewer = rendering.SimpleImageViewer()

#from skimage.transform import rescale, resize, downscale_local_mean
import scipy

def preprocess(newFrame):
	res = newFrame
	im = Image.fromarray(res)
	im = im.convert('L') 
	#im = im.resize(INPUT_SIZE, Image.ANTIALIAS)
	#im = im.resize(INPUT_SIZE, Image.BICUBIC )
	#im = im.resize(INPUT_SIZE, Image.BILINEAR)
	#lastFrame = np.array(im, dtype='uint8')
	lastFrame = scipy.misc.imresize(np.array(im, dtype='uint8'), INPUT_SIZE, interp='bilinear')
	
	lastFrameOrig = None
	if not model_env is None:
		lastFrameOrig = np.array(newFrame, dtype='uint8')
	return lastFrame, lastFrameOrig
	
	
env = gym.make(args.game + 'NoFrameskip-v0')
ACTION_COUNT = env.action_space.n

#env = gym.make('Freeway-v0')
agent = RandomAgent(env.action_space)

def newGame():
	global ob, reward, step_count, episode_reward, lastFrame, lastFrameOrig, lastOb, lastObOrig
	ob = env.reset()
	noop_count = np.random.random_integers(0, NO_OP_MAX)
	for I in range(noop_count):
		ob, _, done, _ = env.step(0)
		if done:
			print('Game terminated during warm up')
	reward = 0
	step_count = 0
	episode_reward = 0
	lastFrame, lastFrameOrig = preprocess(ob)
	lastOb = None
	lastObOrig = None

def gameStep(action):
	tot_reward = 0
	obs = []
	for I in range(ACTION_REPEAT):
		ob, reward, done, _ = env.step(action)
		obs.append(ob)
		tot_reward += reward
		if done:
			break
	obs = obs[-2:]
	ob = np.array(obs)
	ob = np.max(ob, 0)
	return ob, tot_reward, done


if K.backend() == 'tensorflow':
	import tensorflow as tf
elif K.backend() == 'theano':
	from theano import tensor as T
	
if not args.cpu:
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto(log_device_placement=False)
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	config.gpu_options.allow_growth = True
	config.gpu_options.visible_device_list = args.gpu
	set_session(tf.Session(config=config))

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
		
def my_uniform(seed=None):
    return keras.initializers.VarianceScaling(scale=0.3333333,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)

my_optimizer = MyRMSprop(lr=LEARNING_RATE, rho1=GRADIENT_MOMENTUM, rho2=SQUARED_GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT, print_layer=9 if not args.torch_compare_path is None else -1)
my_optimizer_critic = MyRMSprop(lr=LEARNING_RATE, rho1=GRADIENT_MOMENTUM, rho2=SQUARED_GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT, print_layer=9 if not args.torch_compare_path is None else -1)

def my_mean(x, ACTION_COUNT):
	x = K.mean(x, axis=1, keepdims=True)
	x = K.tile(x, (1,ACTION_COUNT))
	return x

def get_model():
	
	if not args.actor_critic:
		input_shape=(AGENT_HISTORY_LENGTH,) + INPUT_SIZE
		input = Input(shape=input_shape, name='observation')
		x = Permute((2,3,1))(input)
		x = ZeroPadding2D(padding=((1,0),(1,0)), name='layer1_padding')(x)
		x = Conv2D(filters=32,kernel_size=8,strides=4,padding="valid",activation="relu", kernel_initializer=my_uniform(), name='layer1')(x)
		x = Conv2D(filters=64,kernel_size=4,strides=2,padding="valid",activation="relu", kernel_initializer=my_uniform(), name='layer2')(x)
		x = Conv2D(filters=64,kernel_size=3,strides=1,padding="valid",activation="relu", kernel_initializer=my_uniform(), name='layer3')(x)
		x = Permute((3,1,2))(x)
		x = Flatten()(x)
		if not args.dueling_network:
			x = Dense(512,activation="relu", kernel_initializer=my_uniform())(x)
			y = Dense(ACTION_COUNT, kernel_initializer=my_uniform())(x)
		else:
			xv = Dense(512,activation="relu", kernel_initializer=my_uniform())(x)
			xa = Dense(512,activation="relu", kernel_initializer=my_uniform())(x)
			v = Dense(1, kernel_initializer=my_uniform())(xv) #,activation="relu"
			a = Dense(ACTION_COUNT, kernel_initializer=my_uniform())(xa) #,activation="relu"
			ma = Lambda(my_mean, arguments={'ACTION_COUNT': ACTION_COUNT})(a)
			y1 = Add()([v, a])
			y = Subtract()([y1, ma])
		model = Model(inputs=[input], outputs=[y])
		
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss=huber_loss)
		model.compile(optimizer=my_optimizer,loss=huber_loss) #
		#model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),loss='mse')
	else:
		input_shape= INPUT_SIZE + (AGENT_HISTORY_LENGTH,)
		input = Input(shape=input_shape, name='observation')
		x = input
		x = Conv2D(filters=16,kernel_size=8,strides=4,padding="valid",activation="relu", name='layer1')(x)
		x = Conv2D(filters=32,kernel_size=4,strides=2,padding="valid",activation="relu", name='layer2')(x)
		#x = Permute((3,1,2))(x)
		x = Flatten()(x)
		x = Dense(256,activation="relu")(x)
		pi = Dense(ACTION_COUNT,activation="softmax")(x)
		V = Dense(1)(x)
		#a_t = K.placeholder(shape=(None, ACTION_COUNT))
		#R = K.placeholder(shape=(None, 1))
		#A = R - V
		#def get_ac_loss(pi, a_t, A):
		#	def loss_actor(y_true, y_pred):
		#		Lpi = K.log(K.sum(y_pred*a_t)) * K.stop_gradient(A)
		#		LH = K.sum(y_pred * K.log(y_pred))
		#		L = Lpi + 0.01 * LH
		#		return L
		#	def loss_critic(y_true, y_pred):
		#		Lv = 0.5 * K.mean(K.square(y_true - y_pred))
		#		return Lv
		#	return loss_actor, huber_loss #loss_critic
		#actor_loss, critic_loss = get_ac_loss(pi, a_t, A)

		def actor_optimizer():
			a_t = K.placeholder(shape=[None, ACTION_COUNT])
			A = K.placeholder(shape=(None, ))
			policy = model_actor.output
			Lpi = -K.sum(K.log(K.sum(policy*a_t, axis=1) + 1e-10) * A)
			LH = K.sum(K.sum(pi * K.log(policy + 1e-10), axis=1))
			L = Lpi + 0.01 * LH
			
			optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
			#optimizer = my_optimizer
			updates = optimizer.get_updates(model_actor.trainable_weights, [], L)
			train = K.function([model_actor.input, a_t, A], [L], updates=updates)
			return train
		def critic_optimizer():
			R = K.placeholder(shape=(None,))
			critic = model_critic.output
			critic = K.print_tensor(critic, message='critic: ')
			Lv = K.mean(K.square(R - critic))
			Lv = K.sum(Lv)
			Lv = K.print_tensor(Lv, message='Lv: ')
			optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
			#optimizer = my_optimizer_critic
			updates = optimizer.get_updates(model_critic.trainable_weights, [], Lv)
			train = K.function([model_critic.input, R], [Lv], updates=updates)
			return train		
		#def actor_optimizer():
		#	action = K.placeholder(shape=[None, ACTION_COUNT])
		#	advantages = K.placeholder(shape=[None, ])
        #
		#	policy = model_actor.output
        #
		#	good_prob = K.sum(action * policy, axis=1)
		#	eligibility = K.log(good_prob + 1e-10) * advantages
		#	actor_loss = -K.sum(eligibility)
        #
		#	entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
		#	entropy = K.sum(entropy)
        #
		#	loss = actor_loss + 0.01*entropy
		#	optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
		#	updates = optimizer.get_updates(model_actor.trainable_weights, [], loss)
		#	train = K.function([model_actor.input, action, advantages], [loss], updates=updates)
        #
		#	return train
        
		# make loss function for Value approximation
		#def critic_optimizer():
		#	discounted_reward = K.placeholder(shape=(None, ))
        #
		#	value = model_critic.output
		#	value = K.print_tensor(value, message='critic: ')
		#	loss = K.mean(K.square(discounted_reward - value))
		#	loss = K.print_tensor(loss, message='Lv: ')
        #
		#	optimizer = RMSprop(lr=2.5e-4, rho=0.99, epsilon=0.01)
		#	updates = optimizer.get_updates(model_critic.trainable_weights, [], loss)
		#	train = K.function([model_critic.input, discounted_reward], [loss], updates=updates)
		#	return train		
		model_actor = Model(inputs=[input], outputs=[pi])
		model_critic = Model(inputs=[input], outputs=[V])
		model_actor._make_predict_function()
		model_critic._make_predict_function()
		model = model_actor
		model.critic = model_critic
		model_actor.manual_optimizer = actor_optimizer()
		model_critic.manual_optimizer = critic_optimizer()
		
		#model = Model(inputs=[input], outputs=[pi, V])
		#model.compile(optimizer=my_optimizer,loss=[actor_loss, critic_loss]) 
		#model.my_at = a_t
		#model.my_r = R
		
	model_layer = Model(inputs=model.input, outputs=model.get_layer('layer1').output)
	return model, model_layer


from torch_utils import *
	
def clone_model(model):
	config = model.get_config()
	weights = model.get_weights()
	new_model = Model.from_config(config)
	new_model.set_weights(weights)
	return new_model

model, model_layer = get_model()
model_eval = None
if args.mode == "train":
	model_eval = clone_model(model)

#print(ta, tr, tterm)

np.set_printoptions(precision=4)

if not args.torch_compare_path is None:
	from torch_compare import torch_compare
	torch_compare(args.torch_compare_path, ACTION_COUNT, model, model_eval, MINIBATCH_SIZE,AGENT_HISTORY_LENGTH, INPUT_SIZE, DISCOUNT_FACTOR)

if False:
	model.load_weights('../rlcode/actor.h5')
	model.critic.load_weights('../rlcode/critic.h5')
	history = np.load('../rlcode/history.npy')
	policy = model.predict(history)[0]
	print(policy)
	states = np.load('../rlcode/states.npy')
	actions = np.load('../rlcode/actions.npy')
	advantages = np.load('../rlcode/advantages.npy')
	discounted_rewards = np.load('../rlcode/discounted_rewards.npy')

	loss1 = model.manual_optimizer([states, actions, advantages])
	loss2 = model.critic.manual_optimizer([states, discounted_rewards])
	print('R', discounted_rewards)
	print(loss1, loss2)
	
	sys.exit()
	
if not args.last_weightfile is None:
	model.load_weights(args.last_weightfile)
	
print(model.summary())

STATE_SIZE = 2048
var_loss_coef1 = K.variable(0)
var_loss_coef2 = K.variable(0)
var_loss_coef3 = K.variable(0)

model_env = model_state = model_next_state = model_next_state_auto = model_reward = meanImage = None

if not args.env_model is None:
	model_env, model_state, model_next_state, model_next_state_auto, model_reward = load_model(args.env_model, args.env_weight, args.env_reward_weight, STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, 1, var_loss_coef1, var_loss_coef2, var_loss_coef3)	
	meanImage = np.load(args.env_mean_image)
	print(model_env.summary())

newGame()
done = False

replay_buffer = SequentialMemory(max_size=REPLAY_MEMORY_SIZE)
total_step_count = 0

#REPLAY_START_SIZE = 1000
#FINAL_EXPLORATION_FRAME = 50000
#REPLAY_START_SIZE = 5000
episode_reward = 0
epsilon = INITIAL_EXPLORATION

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def weight_norms(model):
	ws = model.get_weights()
	for w in ws:
		print(w.shape)
		print('AMEAN: {0}, AMAX: {1}, STD: {2}, MIN: {3}, MAX: {4}'.format(np.mean(np.abs(w)),np.max(np.abs(w)), np.std(w), np.min(w), np.max(w) ))
		#print('MIN: {0}, MAX: {1}'.format(np.min(w), np.max(w) ))
	
current_state = None
r_max = 1
losses = []
while True:
	action = agent.act(ob, reward, done)
	#if total_step_count<=FINAL_EXPLORATION_FRAME and args.mode == "train":
	#	action = np.random.choice(range(ACTION_COUNT), p=[0.30, 0.40, 0.30])
	if args.actor_critic:
		if step_count>=AGENT_HISTORY_LENGTH and ( (total_step_count > REPLAY_START_SIZE) or args.mode == "test"):
			lastItems = replay_buffer.getLastItems(AGENT_HISTORY_LENGTH)
			lastFrames = [a['current_state'] for a in lastItems]
			lastFrames = lastFrames[1:]
			lastFrames.append(lastItems[-1]['next_state'])
			prediction = model.predict_on_batch(np.array([lastFrames], dtype='f')/255.0)[0]
			action = np.random.choice(range(ACTION_COUNT), p=prediction)
	else:
		if (total_step_count > REPLAY_START_SIZE) or args.mode == "test":
			epsilon = (INITIAL_EXPLORATION-FINAL_EXPLORATION) * max(FINAL_EXPLORATION_FRAME-total_step_count, 0) / (FINAL_EXPLORATION_FRAME-REPLAY_START_SIZE) + FINAL_EXPLORATION
			if args.mode == "test":
				epsilon = args.test_epsilon
			if epsilon < random.random():
				if step_count>=AGENT_HISTORY_LENGTH:
					lastItems = replay_buffer.getLastItems(AGENT_HISTORY_LENGTH)
					lastFrames = [a['current_state'] for a in lastItems]
					lastFrames = lastFrames[1:]
					lastFrames.append(lastItems[-1]['next_state'])
					prediction = model.predict_on_batch(np.array([lastFrames], dtype='f')/255.0)[0]
					# @diff: random tie breaking
					action = np.argmax(prediction)
				

	ob, reward, done = gameStep(action)
	reward = max(MIN_REWARD, reward)
	reward = min(MAX_REWARD, reward)
	episode_reward = episode_reward + reward
	
#	print(type(ob))

	lastOb = lastFrame
	lastObOrig = lastFrameOrig
	lastFrame, lastFrameOrig = preprocess(ob)
	
	#if step_count >= 100 and step_count<110:
	#	im2 = Image.fromarray(lastFrame)
	#	im2.save('test-image-my{0}.png'.format(step_count))
	
	if total_step_count%args.render_step == 0 and ENABLE_RENDER:
		viewer.imshow(ob)
	
	#replay_buffer.append(lastOb, action, lastFrame, r, done, None, None)
	replay_buffer.append(lastOb, action, lastFrame, reward, done, None if model_env is None else lastObOrig, None if model_env is None else lastFrameOrig)
	
	if step_count>=AGENT_HISTORY_LENGTH and args.mode == "train":
		if total_step_count % UPDATE_FREQUENCY == 0 and total_step_count>REPLAY_START_SIZE:
			samples = random.sample(xrange(len(replay_buffer)), MINIBATCH_SIZE)
			current_state = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH)+INPUT_SIZE, dtype='f')
			next_state = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH)+INPUT_SIZE, dtype='f')
			rewards = np.zeros((MINIBATCH_SIZE,), dtype='f')
			reward_action = np.zeros((MINIBATCH_SIZE,ACTION_COUNT), dtype='f')
			index_map = range(MINIBATCH_SIZE)
				
			for I in xrange(MINIBATCH_SIZE):
				#sample = replay_buffer[samples[I]]
				lastItems = replay_buffer.getItems(samples[I], AGENT_HISTORY_LENGTH)
				lastFrames = [a['current_state'] for a in lastItems]
				nextFrame = lastItems[-1]['next_state']
				lastFrames.append(nextFrame)
				current_state[I] = np.array(lastFrames[:-1], dtype='f')/255.0
				next_state[I] = np.array(lastFrames[1:], dtype='f')/255.0  
				reward_action[I] = np.array(to_categorical(lastItems[-1]['action'],num_classes=ACTION_COUNT), dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT))
			if not model_env is None:
				current_obs = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*3)+(210,160), dtype='f')
				next_obs = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*3)+(210,160), dtype='f')
				actions = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*ACTION_COUNT), dtype='f')
				for I in xrange(MINIBATCH_SIZE):
					lastItems = replay_buffer.getItems(samples[I], AGENT_HISTORY_LENGTH)
					lastFramesOrig = [a['current_state_orig'] for a in lastItems]
					lastFramesOrig.append(lastItems[-1]['next_state_orig'])
					current_obs[I] = get_obs_input(lastFramesOrig[:-1], meanImage)
					next_obs[I] = get_obs_input(lastFramesOrig[1:], meanImage)
					actions[I,:] = np.array([to_categorical(a['action'],num_classes=ACTION_COUNT) for a in lastItems], dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT*AGENT_HISTORY_LENGTH))
				
				prediction = my_predict(model_env, current_obs, next_obs, actions)
				for I in xrange(MINIBATCH_SIZE):
					tmp1 = prediction_to_image(prediction[I], meanImage)
					tmp1, _ = preprocess(tmp1)
					next_state[I,-1] = np.array(tmp1, dtype='f')/255.0  
				
			if not args.actor_critic:
				target = model.predict(current_state)
			else:
				target = model.critic.predict(current_state)
				discounted_return = np.zeros((MINIBATCH_SIZE,), dtype='f')
			if not args.actor_critic:
				next_value = model_eval.predict(next_state)
				if args.double_dqn:
					next_best_res = model.predict(next_state)
					best_acts = np.argmax(next_best_res, axis=1)
				else:
					best_acts = np.argmax(next_value, axis=1)
			else:
				next_value = model.critic.predict(next_state)
				
			#if total_step_count % 3000 == 0:
			#	print(next_value)
			for I in xrange(MINIBATCH_SIZE):
				lastItems = replay_buffer.getItems(samples[I], AGENT_HISTORY_LENGTH)
				transition = lastItems[-1]
				action = transition['action']
				reward = transition['reward']
				if not args.actor_critic:
					if transition['done']:
						target[I,action] = reward
					else:
						#target[I,action] = reward + DISCOUNT_FACTOR * np.max(next_value[I,:]) #index_map[I]
						target[I,action] = reward + DISCOUNT_FACTOR * next_value[I,best_acts[I]]   #after double DQN
				else:
					if transition['done']:
						discounted_return[I] = reward
					else:
						discounted_return[I] = reward + DISCOUNT_FACTOR * next_value[I]
					
			if not args.actor_critic:
				res = model.train_on_batch(current_state, target)
				print res
			else:
				target = np.reshape(target, (target.shape[0],))
				loss1 = model.manual_optimizer([current_state, reward_action, discounted_return - target])
				loss2 = model.critic.manual_optimizer([current_state, discounted_return])
				print('R', discounted_return)
				print(loss1, loss2)
				loss1 = loss1[0]
				loss2 = loss2[0]
				res = [loss1 + loss2, loss1, loss2]
				#print res
			#print(res)
			losses.append(res)
	if total_step_count%1000 == 0 and args.mode == "train":
		if not args.output_dir is None:
			np.savetxt(args.output_dir + '/losses.txt', losses)
				
	if total_step_count % TARGET_NETWORK_UPDATE_FREQUENCY == 1 and args.mode == "train":
		model_eval.set_weights(model.get_weights())
	if total_step_count % SAVE_FREQ == 0 and args.mode == "train":
		if not args.output_dir is None:
			model.save_weights(args.output_dir + '/weights_{0}.h5'.format(total_step_count))
	total_step_count = total_step_count + 1
	step_count = step_count + 1
	
	#weight_norms(model)
	
	if done:
		#print('new episode, previous reward: ', episode_reward)
		print('Episode end. Step Count: {0}, Total Step Count: {1}, Total reward: {2}'.format(step_count, total_step_count, episode_reward))
		newGame()

#time.sleep(5) 

