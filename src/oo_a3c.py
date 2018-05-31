import argparse

parser = argparse.ArgumentParser(description='A3C Training')
parser.add_argument('game', type=str, default='Breakout', help='Gym game name')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test')
parser.add_argument('--output-dir', type=str, default=None, help='output directory')
parser.add_argument('--double-dqn', type=bool, default=False, help='Use double dqn')
parser.add_argument('--dueling-dqn', type=bool, default=False, help='Dueling dqn')
parser.add_argument('--logdir', type=str, default=None, help='Logdir for tensorboard')
parser.add_argument('--thread-count', type=int, default=2, help='Number of threads')
parser.add_argument('--enable-render', type=bool, default=False, help='enable render')
parser.add_argument('--render-step', type=int, default=4, help='Render at this steps')
parser.add_argument('--nstep', type=int, default=5, help='step count for n-step q learning')
parser.add_argument('--atari', type=bool, default=False, help='true if env is atari game')
parser.add_argument('--model', type=str, default='A3CModel', help='class name for q-model')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate')
parser.add_argument('--target-network-update', type=int, default=10000, help='target network update feq')
parser.add_argument('--egreedy-props', type=float, nargs='*', default=[0.4, 0.3, 0.3], help='multiple egreedy props')
parser.add_argument('--egreedy-final', type=float, nargs='*', default=[0.1, 0.01, 0.5], help='multiple egreedy final exploration')
parser.add_argument('--egreedy-final-step', type=int, nargs='*', default=1e6, help='multiple egreedy final step')
parser.add_argument('--egreedy-decay', type=float, nargs='*', default=[1,1,1], help='exponential decay rate for egreedy')
parser.add_argument('--env-transforms', type=str, nargs='*', default=[], help='apply the environment transforms')
parser.add_argument('--update-frequency', type=int, default=4, help='training update frequency')
parser.add_argument('--replay-buffer-size', type=int, default=int(1e6), help='the number of transitions in replay buffer')
parser.add_argument('--replay-start-size', type=int, default=int(50000), help='replay start size')
parser.add_argument('--batch-size', type=int, default=int(32), help='batch size')
parser.add_argument('--max-step', type=int, default=int(1e10), help='max step')
parser.add_argument('--save-interval', type=int, default=50000, help='save interval')

args = parser.parse_args()

from algo.a3c import run_a3c

arguments = vars(args)

import numpy as np

stats = run_a3c(**arguments)

print(stats)

all_pars = {
	'learning_rate': [0.05, 0.02, 0.01, 0.0075, 0.005, 0.001],
	'egreedy_decay': [0.99, 0.995, 0.998, 0.999],
	'nstep': [1, 2, 5]
}

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def coordinateDescent(arguments):
	best_pars = arguments.copy()
	all_stats = []
	aver_rewards = []
	all_params = []
	best_aver_reward = -1000000

	for I in range(2):
		for par_name, par_values in all_pars.iteritems():
			for par_value in par_values:
				curr_pars = best_pars.copy()
				curr_pars[par_name] = par_value
				logstr = ''
				for key in all_pars:
					logstr += key + '_' + str(curr_pars[key]) + '_'
				curr_pars['logdir'] = arguments['logdir'] + '/' + logstr
				print(curr_pars, logstr)
				stats = run_a3c(**curr_pars)
				if stats is None:
					exit()
				tot_reward = np.array([a[1] for a in stats['reward']]).sum()
				aver_reward = tot_reward / stats['reward'][-1][0]
				all_stats.append(stats)
				aver_rewards.append(aver_reward)
				all_params.append(curr_pars)
				if aver_reward > best_aver_reward:
					best_aver_reward = aver_reward
					best_pars = curr_pars.copy()
			
	print(all_params)
	print(all_stats)
	print(aver_rewards)
	print(best_pars)
	print(best_aver_reward)

#ENABLE_RENDER = args.enable_render
#
#if not args.output_dir is None:
#	import os, sys
#	if not os.path.exists(args.output_dir):
#		os.makedirs(args.output_dir)
#if args.output_dir is None and args.logdir is not None:
#	args.output_dir = args.logdir
#		
#from PIL import Image
#
#from envs.gym_env import gym_env
#from envs.env_transform import *
#from envs.env import GridEnv
#from utils.preprocess import *
#from runner.runner import Runner, RunnerListener
#from agents.agent import *
#from utils.memory import ReplayBuffer, NStepBuffer
#from nets.net import *
#import tensorflow as tf
#import keras.backend as K
#
#import threading
#from threading import Lock
#from utils.stopable_thread import StoppableThread
#
#if ENABLE_RENDER:
#	from gym.envs.classic_control import rendering
#	viewer = rendering.SimpleImageViewer()
#
#init_nn_library(False, "1")
#
#if args.atari:
#	env = gym_env(args.game + 'NoFrameskip-v0')
#else:
#	if args.game == "Grid":
#		env = GridEnv()
#	else:
#		env = gym_env(args.game)
#
#modelOps = DqnOps(env.action_count)
#modelOps.dueling_network = args.dueling_dqn
#
#if args.atari:
#	#model = A3CModel(modelOps)
#	SAVE_FREQ = 100000
#	#modelOps.LEARNING_RATE = 0.00025
#	#target_network_update_freq = 10000
#else:
#	if env.observation_space.__class__.__name__ is 'Discrete':
#		modelOps.INPUT_SIZE = env.observation_space.n
#	else:
#		modelOps.INPUT_SIZE = env.observation_space.shape
#	modelOps.AGENT_HISTORY_LENGTH = 1
#	#modelOps.LEARNING_RATE = 0.2
#	#model = TabularQModel(modelOps)
#	#target_network_update_freq = 100
#	SAVE_FREQ = 1e10
#
#modelOps.LEARNING_RATE = args.learning_rate
#target_network_update_freq = args.target_network_update
#	
#model = globals()[args.model](modelOps)
#
#T = 0
#tLock = Lock()
#
#sess = tf.InteractiveSession()
#K.set_session(sess) 
#
#model_eval = model.clone()
#
#sess.run(tf.global_variables_initializer())
#
#graph = tf.get_default_graph()
#
#class AgentThread(StoppableThread, RunnerListener):
#	def __init__(self, threadId, sess, graph):
#		StoppableThread.__init__(self)
#		self.threadId = threadId
#		self.sess = sess
#		self.graph = graph
#		with self.graph.as_default():
#			if args.atari:
#				env = gym_env(args.game + 'NoFrameskip-v0')
#				env = WarmUp(env, min_step=0, max_step=30)
#				env = ActionRepeat(env, 4)
#				proproc = PreProPipeline([GrayPrePro(), ResizePrePro(modelOps.INPUT_SIZE)])
#				rewproc = PreProPipeline([RewardClipper(-1, 1)])
#				#q_model = A3CModel(modelOps)
#			else:
#				if args.game == "Grid":
#					env = GridEnv()
#				else:
#					env = gym_env(args.game)
#				proproc = None
#				rewproc = None
#				#q_model = TabularQModel(modelOps)
#			for trans in args.env_transforms:
#				env = globals()[trans](env)
#			q_model = globals()[args.model](modelOps)
#
#			q_model.model_update = model.model
#			q_model.set_weights(model.get_weights())
#			summary_writer = tf.summary.FileWriter(args.logdir + '/thread-'+str(threadId), K.get_session().graph) if not args.logdir is None else None
#
#			agentOps = DqnAgentOps()
#			agentOps.double_dqn = args.double_dqn
#			agentOps.REPLAY_START_SIZE = 1
#			#agentOps.INITIAL_EXPLORATION = 0
#			agentOps.TARGET_NETWORK_UPDATE_FREQUENCY = 1e10
#
#			#replay_buffer = ReplayBuffer(int(1e6), 4, 4, agentOps.REPLAY_START_SIZE, 32)
#			if args.nstep > 0:
#				replay_buffer = NStepBuffer(modelOps.AGENT_HISTORY_LENGTH, args.nstep)
#			else:
#				replay_buffer = ReplayBuffer(args.replay_buffer_size, modelOps.AGENT_HISTORY_LENGTH, args.update_frequency, args.replay_start_size, args.batch_size)
#
#			agent = DqnAgent(env.action_space, q_model, replay_buffer, rewproc, agentOps, summary_writer, model_eval=model_eval) #
#
#			egreedyOps = EGreedyOps()
#			egreedyOps.REPLAY_START_SIZE = replay_buffer.REPLAY_START_SIZE
#			egreedyOps.FINAL_EXPLORATION_FRAME = int(args.egreedy_final_step / args.thread_count)				
#
#			if args.egreedy_decay<1:
#				egreedyOps.DECAY = args.egreedy_decay
#				egreedyAgent = EGreedyAgentExp(env.action_space, egreedyOps, agent)
#			else:
#				egreedyAgent = MultiEGreedyAgent(env.action_space, egreedyOps, agent, args.egreedy_props, args.egreedy_final)
#			
#			self.runner = Runner(env, egreedyAgent, proproc, modelOps.AGENT_HISTORY_LENGTH)
#			self.runner.listen(replay_buffer, proproc)
#			self.runner.listen(agent, None)
#			self.runner.listen(egreedyAgent, None)
#			self.runner.listen(self, proproc)
#		pass
#	def run(self):
#		with self.graph.as_default():
#			self.runner.run()
#	def on_step(self, ob, action, next_ob, reward, done):
#		global T
#		global model, model_eval
#		with tLock:
#			T = T + 1
#		#if T % 1000 == 0:
#		#	print('STEP', T)
#		if T % target_network_update_freq == 0:
#			print('CLONE TARGET: ' + str(T))
#			model_eval.set_weights(model.get_weights())
#			for agent in agents:
#				agent.model_eval = model_eval
#		if T%args.render_step == 0 and ENABLE_RENDER:
#			viewer.imshow(np.repeat(np.reshape(ob, ob.shape + (1,)), 3, axis=2))
#		if T % SAVE_FREQ == 0 and args.mode == "train":
#			if not args.output_dir is None:
#				model.model.save_weights(args.output_dir + '/weights_{0}.h5'.format(T))
#		#print(T)
#	def stop(self):
#		super(AgentThread, self).stop()
#		self.runner.stop()
#
#import time
#agents = []
#for I in range(args.thread_count):
#	agent = AgentThread(I, sess, graph)
#	agents.append(agent)
#
#print("NOW START THE AGENTS ****************")
#for agent in agents:
#	time.sleep(1)
#	agent.start()
#	#self.default_graph.finalize()
#	
#while True:
#	try:
#		print('HEART BEAT')
#		time.sleep(10)
#	except KeyboardInterrupt: 
#		print('KILL AGENTS')
#		for agent in agents:
#			agent.stop()
#		print('TIME TO STOP')
#		exit()
#