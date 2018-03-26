from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Learn Model.')
parser.add_argument('weight_idx', type=int, help='weight idx. actual file is 125000 times this value')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('model', type=str, default="env", help='Network mode (acvp or env)')
parser.add_argument('last_weightfile', nargs='?', type=str, default="", help='last weight file')
parser.add_argument('--mode', type=str, default="train", help='operation mode (train or test)')
parser.add_argument('--lossa', nargs=1, type=float, default=1.0, help='Loss coef of autoencoder')
parser.add_argument('--loss1', nargs=2, type=float, default=[0, 0], help='State regularizer')
parser.add_argument('--loss2', nargs=2, type=float, default=[0, 0], help='Autoencoder vs next prediction regularizer')
parser.add_argument('--loss3', nargs=2, type=float, default=[0, 0], help='Action prediction regularizer')
parser.add_argument('--max-episode', type=int, help='Maximum episode number')
parser.add_argument('--enable-render', type=bool, default=False, help='Enable render')
parser.add_argument('--render-step', type=int, default=4, help='Render at this steps')
parser.add_argument('--mean-image', type=str, default=None, help='Load mean image')
parser.add_argument('--gpu', type=str, default="1", help='Which gpu to use')
parser.add_argument('--dqn-weight', type=str, default=None, help='dqn weight file. overrides weight_idx')
parser.add_argument('--graph-file', type=str, default=None, help='filename to save graph visualization')
parser.add_argument('--train-target', type=str, default="model", help='what to train: model or reward')
parser.add_argument('--reward-weight', type=str, default=None, help='weight file for reward estimation')
parser.add_argument('--state-multiplier', type=float, default=1.0, help='state is multiplied by this value for reward estimation')
parser.add_argument('--compare-models', type=str, nargs='*', default=None, help='Other models to compare')
parser.add_argument('--save-frames', type=str, default=None, help='Save rgb frames to image files')


args = parser.parse_args()
#print(args.weight_start)
#print(args.weight_end)
#print(args.weight_inc)
#quit()

import os
import sys

print(' '.join(sys.argv))

ENABLE_RENDER = args.enable_render

total_step_count = 0

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Permute, Activation, concatenate, Input, Reshape, Conv2DTranspose, ZeroPadding2D, Cropping2D, Multiply, Layer, Lambda
from keras.initializers import RandomUniform
from keras.objectives import mean_squared_error
from keras.utils import to_categorical
import keras.backend as K
import random

from PIL import Image, ImageDraw
import numpy as np

from utils import get_activations
from models import get_dqn_model
from models import load_model
from models import my_predict
from utils import toRGBImage
from utils import prediction_to_image
from utils import draw_reward
from utils import get_obs_input

if ENABLE_RENDER:
	from gym.envs.classic_control import rendering
from memory import SequentialMemory

INPUT_SIZE=(84,84)
MINIBATCH_SIZE=32
REPLAY_MEMORY_SIZE=int(1e4)
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

MEAN_IMAGE_CALC_FRAME = 5000

REPLAY_START_SIZE = 5000

if ENABLE_RENDER:
	viewer = rendering.SimpleImageViewer()

def preprocess(newFrame):
	res = newFrame
	im = Image.fromarray(res)
	im = im.convert('L') 
	im = im.resize(INPUT_SIZE, Image.ANTIALIAS)
	lastFrame = np.array(im, dtype='uint8')
	lastFrameOrig = np.array(newFrame, dtype='uint8')
	return lastFrame, lastFrameOrig

import gym
env = gym.make('FreewayNoFrameskip-v0')
ACTION_COUNT = env.action_space.n

def newGame():
	global ob, reward, step_count, episode_reward, lastFrame, lastFrameOrig, lastOb, lastObOrig
	ob = env.reset()
	for I in range(NO_OP_MAX):
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
	for I in range(ACTION_REPEAT):
		ob, reward, done, _ = env.step(action)
		tot_reward += reward
		if done:
			break
	return ob, tot_reward, done

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = args.gpu
set_session(tf.Session(config=config))


STATE_SIZE = 2048

var_loss_coef1 = K.variable(0)
var_loss_coef2 = K.variable(0)
var_loss_coef3 = K.variable(0)

	
model = get_dqn_model(AGENT_HISTORY_LENGTH, INPUT_SIZE, ACTION_COUNT, LEARNING_RATE, GRADIENT_MOMENTUM, SQUARED_GRADIENT_MOMENTUM, MIN_SQUARED_GRADIENT)
max_episode = 10000

if args.max_episode:
	max_episode = args.max_episode


model_env, model_state, model_next_state, model_next_state_auto, model_reward = load_model(args.model, args.last_weightfile, args.reward_weight, STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, args.lossa, var_loss_coef1, var_loss_coef2, var_loss_coef3)	

models_compare = []
if not args.compare_models is None:
	for I in range(int(len(args.compare_models)/4)):
		models_compare.append(load_model(args.compare_models[4*I], args.compare_models[4*I+1], args.compare_models[4*I+2], args.lossa, STATE_SIZE, ACTION_COUNT, AGENT_HISTORY_LENGTH, var_loss_coef1, var_loss_coef2, var_loss_coef3) + (float(args.compare_models[4*I+3]),))
	
print(models_compare)

print(model_env.summary())

if not args.graph_file is None:
	from keras.utils import plot_model
	plot_model(model_env, to_file=args.graph_file, show_shapes=True, show_layer_names=True)

replay_buffer = SequentialMemory(max_size=REPLAY_MEMORY_SIZE)

if args.dqn_weight is None:
	model.load_weights('run2/weights_{0}.h5'.format(args.weight_idx * 125000))
else:
	model.load_weights(args.dqn_weight)
total_reward = 0.0
newGame()
epNo = 0

if args.mode == "train":
	loss_log = open(args.output_dir + '/loss.txt', "a")

if not args.mean_image is None:
	meanImage = np.load(args.mean_image)
else:
	meanImage = np.zeros((210,160,3), dtype="f")

while True:
	action = env.action_space.sample()
	if 0.05 < random.random():
		if step_count>=AGENT_HISTORY_LENGTH+1:
			lastItems = replay_buffer.getLastItems(AGENT_HISTORY_LENGTH)
			lastFrames = [a['next_state'] for a in lastItems]
			prediction = model.predict_on_batch(np.array([lastFrames], dtype='f')/255.0)[0]
			action = np.argmax(prediction)
	
	ob, reward, done = gameStep(action)
	total_reward = total_reward + reward
	episode_reward = episode_reward + reward
	total_step_count = total_step_count + 1
	step_count = step_count + 1
	
	lastOb = lastFrame
	lastObOrig = lastFrameOrig
	lastFrame, lastFrameOrig = preprocess(ob)
	
	if total_step_count==args.loss1[0]:
		K.set_value(var_loss_coef1, args.loss1[1])
	if total_step_count==args.loss2[0]:
		K.set_value(var_loss_coef2, args.loss2[1])
	if total_step_count==args.loss3[0]:
		K.set_value(var_loss_coef3, args.loss3[1])
	
	if args.mean_image is None and total_step_count <= MEAN_IMAGE_CALC_FRAME:
		meanImage = meanImage + lastFrameOrig
		if total_step_count == MEAN_IMAGE_CALC_FRAME:
			meanImage = meanImage / MEAN_IMAGE_CALC_FRAME
			np.save(args.output_dir + '/mean_image', meanImage)
	
	r = max(MIN_REWARD, reward)
	r = min(MAX_REWARD, reward)
	replay_buffer.append(lastOb, action, lastFrame, r, done, lastObOrig, lastFrameOrig)
	if step_count>=AGENT_HISTORY_LENGTH+2:
		if total_step_count%args.render_step == 0 and ENABLE_RENDER:
			lastItems = replay_buffer.getLastItems(AGENT_HISTORY_LENGTH)
			lastFramesOrig = [a['current_state_orig'] for a in lastItems]
			lastFramesOrig.append(lastItems[-1]['next_state_orig'])
			lastActions = np.array([to_categorical(a['action'],num_classes=ACTION_COUNT) for a in lastItems], dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT*AGENT_HISTORY_LENGTH))
			
			netin = get_obs_input([lastFramesOrig[:-1]], meanImage)
			next_netin = get_obs_input([lastFramesOrig[1:]], meanImage)
			prediction = my_predict(model_env, netin, next_netin, lastActions)
			predImage = prediction_to_image(prediction, meanImage)
			
			estimated_reward = 0
			lastAction = np.array(to_categorical(lastItems[-1]['action'],num_classes=ACTION_COUNT), dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT))
			if not model_reward is None:
				tmp_state = model_state.predict_on_batch([netin])
				print(tmp_state)
				print('state norm: ', np.linalg.norm(tmp_state), np.linalg.norm(tmp_state/120))
				estimated_reward = model_reward.predict_on_batch([tmp_state*args.state_multiplier, lastAction])
				print("Reward: ", estimated_reward, lastItems[-1]['reward'])
				predImage = draw_reward(predImage, estimated_reward)
			beforeImage = lastFramesOrig[-2]
			lastImage = lastFramesOrig[-1]
			lastImage = draw_reward(lastImage, lastItems[-1]['reward'])

			images = (beforeImage, lastImage, predImage)
			print(lastImage.shape, predImage.shape)
			for tmp_model in models_compare:
				tmp_prediction = my_predict(tmp_model[0], netin, next_netin, lastActions)
				tmp_predImage = prediction_to_image(tmp_prediction, meanImage)
				if not model_reward is None:
					tmp_state = tmp_model[1].predict_on_batch([netin])
					tmp_estimated_reward = tmp_model[4].predict_on_batch([tmp_state*tmp_model[5], lastAction])
					tmp_predImage = draw_reward(tmp_predImage, tmp_estimated_reward)
				
				images = images + (tmp_predImage,)
			
			bothImage = np.concatenate(images, axis=1)
			if not args.save_frames is None:
				if not os.path.exists(args.save_frames):
					os.makedirs(args.save_frames)
				im = Image.fromarray(bothImage)
				im.save(args.save_frames + "/frame_" + str(step_count).zfill(5) + ".png")
			viewer.imshow(bothImage)

		if args.mode=="train" and total_step_count % UPDATE_FREQUENCY == 0 and total_step_count>REPLAY_START_SIZE:
			#samples = random.sample(xrange(len(replay_buffer)), MINIBATCH_SIZE)
			samples = replay_buffer.samples(MINIBATCH_SIZE, True if args.train_target == "reward" else False)
			current_obs = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*3)+(210,160), dtype='f')
			next_obs = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*3)+(210,160), dtype='f')
			real_next_obs = np.zeros((MINIBATCH_SIZE,)+(210,160,3), dtype='f')
			real_current_obs = np.zeros((MINIBATCH_SIZE,)+(210,160,3), dtype='f')
			actions = np.zeros((MINIBATCH_SIZE,AGENT_HISTORY_LENGTH*ACTION_COUNT), dtype='f')
			rewards = np.zeros((MINIBATCH_SIZE,), dtype='f')
			reward_action = np.zeros((MINIBATCH_SIZE,ACTION_COUNT), dtype='f')
			for I in xrange(MINIBATCH_SIZE):
				lastItems = replay_buffer.getItems(samples[I], AGENT_HISTORY_LENGTH)
				lastFramesOrig = [a['current_state_orig'] for a in lastItems]
				lastFramesOrig.append(lastItems[-1]['next_state_orig'])
				current_obs[I] = get_obs_input(lastFramesOrig[:-1], meanImage)
				next_obs[I] = get_obs_input(lastFramesOrig[1:], meanImage)
				real_next_obs[I] = np.array(lastFramesOrig[-1], dtype='f')/255.0 - meanImage/255.0
				real_current_obs[I] = np.array(lastFramesOrig[-2], dtype='f')/255.0 - meanImage/255.0
				actions[I,:] = np.array([to_categorical(a['action'],num_classes=ACTION_COUNT) for a in lastItems], dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT*AGENT_HISTORY_LENGTH))
				rewards[I] = lastItems[-1]['reward']
				reward_action[I] = np.array(to_categorical(lastItems[-1]['action'],num_classes=ACTION_COUNT), dtype='f').flatten().squeeze().reshape((1, ACTION_COUNT))
			if total_step_count < REPLAY_START_SIZE + 10:
				print(model_env.metrics_names)
			if args.train_target == "model":
				if args.model == "env":
					res = model_env.train_on_batch([current_obs, next_obs, actions], [real_next_obs, real_current_obs, np.zeros((MINIBATCH_SIZE,1)), np.zeros((MINIBATCH_SIZE,1)), np.zeros((MINIBATCH_SIZE,1))]) #
				else:
					res = model_env.train_on_batch([current_obs, actions], [real_next_obs]) 
			elif args.train_target == "reward":
				state = model_state.predict_on_batch([current_obs])
				res = model_reward.train_on_batch([state*args.state_multiplier, reward_action], [rewards])
			print(total_step_count, res)
			if not isinstance(res,list):
				res = [res]
			loss_log.write('\t'.join([str(x) for x in [total_step_count] + res]) + '\n')
			loss_log.flush()
	if total_step_count % SAVE_FREQ == 0 and args.mode == "train":
		if args.train_target == "model":
			model_env.save_weights(args.output_dir + '/weights_{0}.h5'.format(total_step_count))
		elif args.train_target == "reward":
			model_reward.save_weights(args.output_dir + '/weights_{0}.h5'.format(total_step_count))
	
	if done:
		print('Episode end. Step Count: {0}, Total Step Count: {1}, Total reward: {2}'.format(step_count, total_step_count, episode_reward))
		epNo = epNo + 1
		if epNo<max_episode:
			newGame()
		else:
			break
			#quit()
print("AverageReward: {0}".format(total_reward/max_episode))
if args.mode == "train":
	loss_log.close()