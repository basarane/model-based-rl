import argparse

parser = argparse.ArgumentParser(description='A3C Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--output-dir', type=str, default=None, help='output directory')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--thread-count', type=int, default=2, help='Number of threads')
parser.add_argument('--enable-render', type=bool, default=False, help='enable render')
parser.add_argument('--render-step', type=int, default=4, help='Render at this steps')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--nstep', type=int, default=5, help='step count for n-step q learning')
parser.add_argument('--is-test', type=bool, default=False, help='true if tested with discrete state (grid worlds)')

args = parser.parse_args()

is_test = args.is_test

ENABLE_RENDER = args.enable_render

if not args.output_dir is None:
	import os, sys
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
if args.output_dir is None and args.logdir is not None:
	args.output_dir = args.logdir
		
from PIL import Image

from envs.gym_env import gym_env
from envs.env_transform import WarmUp, ActionRepeat, ObservationStack
from envs.env import GridEnv
from utils.preprocess import *
from runner.runner import Runner, RunnerListener
from agents.agent import DqnAgent, DqnAgentOps, EGreedyOps, EGreedyAgent, MultiEGreedyAgent
from utils.memory import ReplayBuffer, NStepBuffer
from nets.net import A3CModel, DQNModel, DqnOps, init_nn_library, TabularQModel
import tensorflow as tf
import keras.backend as K

import threading
from threading import Lock
from utils.stopable_thread import StoppableThread

if ENABLE_RENDER:
	from gym.envs.classic_control import rendering
	viewer = rendering.SimpleImageViewer()

init_nn_library(False, "1")

if not is_test:
	env = gym_env(args.game + 'NoFrameskip-v0')
else:
	if args.game == "Grid":
		env = GridEnv()
	else:
		env = gym_env(args.game)

modelOps = DqnOps(env.action_count)
modelOps.dueling_network = args.dueling_dqn

if not is_test:
	model = A3CModel(modelOps)
	SAVE_FREQ = 100000
	modelOps.LEARNING_RATE = 0.001
	target_network_update_freq = 10000
else:
	modelOps.INPUT_SIZE = env.observation_space.n
	modelOps.LEARNING_RATE = 0.2
	modelOps.AGENT_HISTORY_LENGTH = 1
	model = TabularQModel(modelOps)
	SAVE_FREQ = 1e10
	target_network_update_freq = 100


T = 0
tLock = Lock()

sess = tf.InteractiveSession()
K.set_session(sess) 

model_eval = model.clone()

sess.run(tf.global_variables_initializer())

graph = tf.get_default_graph()

class AgentThread(StoppableThread, RunnerListener):
	def __init__(self, threadId, sess, graph):
		StoppableThread.__init__(self)
		self.threadId = threadId
		self.sess = sess
		self.graph = graph
		with self.graph.as_default():
		
			if not is_test:
				env = gym_env(args.game + 'NoFrameskip-v0')
				env = WarmUp(env, min_step=0, max_step=30)
				env = ActionRepeat(env, 4)
				proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
				rewproc = PreProPipeline([RewardClipper(-1, 1)])
				q_model = A3CModel(modelOps)
			else:
				if args.game == "Grid":
					env = GridEnv()
				else:
					env = gym_env(args.game)
				proproc = None
				rewproc = None
				q_model = TabularQModel(modelOps)

			q_model.model_update = model.model
			q_model.set_weights(model.get_weights())
			summary_writer = tf.summary.FileWriter(args.logdir + '/thread-'+str(threadId), K.get_session().graph) if not args.logdir is None else None

			agentOps = DqnAgentOps()
			agentOps.double_dqn = args.double_dqn
			agentOps.REPLAY_START_SIZE = 1
			#agentOps.INITIAL_EXPLORATION = 0
			agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = 1e10

			#replay_buffer = ReplayBuffer(int(1e6), 4, 4, agentOps.REPLAY_START_SIZE, 32)
			replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, args.nstep)
			agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer, model_eval=model_eval) #

			egreedyOps = EGreedyOps()
			egreedyOps.REPLAY_START_SIZE = 1
			if not is_test:
				egreedyOps.FINAL_EXPLORATION_FRAME = int(1e6 / args.thread_count)				
			else:
				egreedyOps.FINAL_EXPLORATION_FRAME = 5000
			egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, [0.4, 0.3, 0.3], [0.1, 0.01, 0.5])

			self.runner = Runner(env, egreedyAgent, proproc, modelOps.AGENT_HISTORY_LENGTH)
			self.runner.listen(replay_buffer, proproc)
			self.runner.listen(agent, None)
			self.runner.listen(egreedyAgent, None)
			self.runner.listen(self, proproc)
		pass
	def run(self):
		with self.graph.as_default():
			self.runner.run()
	def on_step(self, ob, action, next_ob, reward, done):
		global T
		global model, model_eval
		with tLock:
			T = T + 1
		#if T % 1000 == 0:
		#	print('STEP', T)
		if T % target_network_update_freq == 0:
			print('CLONE TARGET')
			model_eval.set_weights(model.get_weights())
			for agent in agents:
				agent.model_eval = model_eval
		if T%args.render_step == 0 and ENABLE_RENDER:
			viewer.imshow(np.repeat(np.reshape(ob, ob.shape + (1,)), 3, axis=2))
		if T % SAVE_FREQ == 0 and args.mode == "train":
			if not args.output_dir is None:
				model.model.save_weights(args.output_dir + '/weights_{0}.h5'.format(T))
		#print(T)
	def stop(self):
		super(AgentThread, self).stop()
		self.runner.stop()

import time
agents = []
for I in range(args.thread_count):
	agent = AgentThread(I, sess, graph)
	agents.append(agent)

print("NOW START THE AGENTS ****************")
for agent in agents:
	time.sleep(1)
	agent.start()
	#self.default_graph.finalize()
	
while True:
	try:
		print('HEART BEAT')
		time.sleep(10)
	except KeyboardInterrupt: 
		print('KILL AGENTS')
		for agent in agents:
			agent.stop()
		print('TIME TO STOP')
		exit()
