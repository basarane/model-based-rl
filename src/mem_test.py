import gym
import numpy as np
from PIL import Image
from gym import wrappers
from memory import SequentialMemory
import random
import scipy
import gc
import lz4framed

NO_OP_MAX=30
INPUT_SIZE=(84,84)
REPLAY_MEMORY_SIZE=int(1e6)
ACTION_REPEAT=4
ENABLE_RENDER=False
AGENT_HISTORY_LENGTH = 4
MINIBATCH_SIZE=32

if ENABLE_RENDER:
	from gym.envs.classic_control import rendering
	viewer = rendering.SimpleImageViewer()

class RandomAgent(object):
	def __init__(self, action_space):
		self.action_space = action_space
	def act(self, observation, reward, done):
		return self.action_space.sample()

def preprocess(newFrame):
	res = newFrame
	im = Image.fromarray(res)
	im = im.convert('L') 
	#im = im.resize(INPUT_SIZE, Image.ANTIALIAS)
	#im = im.resize(INPUT_SIZE, Image.BICUBIC )
	im = im.resize(INPUT_SIZE, Image.BILINEAR)
	lastFrame = np.array(im, dtype='uint8')
	#lastFrame = scipy.misc.imresize(np.array(im, dtype='uint8'), INPUT_SIZE, interp='bilinear')
	lastFrameOrig = None
	return lastFrame, lastFrameOrig
	
	
env = gym.make('FreewayNoFrameskip-v0')
ACTION_COUNT = env.action_space.n

#env = gym.make('Freeway-v0')
agent = RandomAgent(env.action_space)

def newGame():
	global ob, reward, step_count, episode_reward, lastFrame, lastFrameOrig, lastOb, lastObOrig, lastFrameCompressed, lastObCompressed
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
	lastObCompressed = None
	lastFrameCompressed = lz4framed.compress(lastFrame)

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
	
import pickle
def serialize_array(a):
	return pickle.dumps(a, protocol=2) # protocol 0 is printable ASCII

def deserialize_array(serialized):
	return pickle.loads(serialized)

newGame()
done = False

replay_buffer = SequentialMemory(max_size=REPLAY_MEMORY_SIZE)
total_step_count = 0

print('GC', gc.isenabled())
#gc.set_debug(True)

while True:
	action = agent.act(ob, reward, done)

	ob, reward, done = gameStep(action)

	lastOb = lastFrame
	lastObCompressed = lastFrameCompressed
	lastObOrig = lastFrameOrig
	lastFrame, lastFrameOrig = preprocess(ob)
	lastFrameCompressed = lz4framed.compress(serialize_array(lastFrame))

	if ENABLE_RENDER:
		tmp1 = deserialize_array(lz4framed.decompress(lastFrameCompressed))
		print(tmp1.shape)
		tmp1 = tmp1.reshape(INPUT_SIZE + (1,))
		tmp1 = np.tile(tmp1, (1,1,3))
		viewer.imshow(tmp1)

	replay_buffer.append(lastObCompressed, action, lastFrameCompressed, reward, done, None, None)
	
	if total_step_count > 1000:
		samples = random.sample(xrange(len(replay_buffer)), MINIBATCH_SIZE)
		for I in xrange(MINIBATCH_SIZE):
			#sample = replay_buffer[samples[I]]
			lastItems = replay_buffer.getItems(samples[I], AGENT_HISTORY_LENGTH)
			lastFrames = [a['current_state'] for a in lastItems]
			nextFrame = lastItems[-1]['next_state']
			print(nextFrame)
			lastFrames.append(nextFrame)
			#print('deserialize')
			lastFrames = [deserialize_array(lz4framed.decompress(a)) for a in lastFrames]
	total_step_count = total_step_count + 1
	step_count = step_count + 1
	if done:
		print('Episode end. Step Count: {0}, Total Step Count: {1}'.format(step_count, total_step_count))
		newGame()
